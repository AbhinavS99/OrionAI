import asyncio
import pickle
import random
import json
import copy
import yaml
from functools import wraps
from enum import Enum
from abc import ABC, abstractmethod
from openai import AzureOpenAI
from collections import defaultdict
from typing import Optional, Any, List, Callable, Dict, Awaitable
from dataclasses import dataclass, field

class GraphOrchestratorException(Exception):
    pass

class DuplicateNodeError(GraphOrchestratorException):
    def __init__(self, node_id: str):
        super().__init__(f"Node with id '{node_id}' already exists.")
        self.node_id = node_id


class EdgeExistsError(GraphOrchestratorException):
    def __init__(self, source_id: str, sink_id: str):
        super().__init__(f"Edge from '{source_id}' to '{sink_id}' already exists.")
        self.source_id, self.sink_id = source_id, sink_id


class NodeNotFoundError(GraphOrchestratorException):
    def __init__(self, node_id: str):
        super().__init__(f"Node '{node_id}' not found in the graph.")
        self.node_id = node_id


class GraphConfigurationError(GraphOrchestratorException):
    def __init__(self, message: str):
        super().__init__(f"Graph configuration error: {message}")


class GraphExecutionError(GraphOrchestratorException):
    def __init__(self, node_id: str, message: str):
        super().__init__(f"Execution failed at node '{node_id}': {message}")
        self.node_id, self.message = node_id, message


class InvalidRoutingFunctionOutput(GraphOrchestratorException):
    def __init__(self, returned_value):
        super().__init__(f"Routing function must return a string, but got {type(returned_value).__name__}: {returned_value}")


class InvalidNodeActionOutput(GraphOrchestratorException):
    def __init__(self, returned_value):
        super().__init__(f"Node action must return a state, but got {type(returned_value).__name__}: {returned_value}")


class InvalidToolMethodOutput(GraphOrchestratorException):
    def __init__(self, returned_value):
        super().__init__(f"Tool method must return a state, but got {type(returned_value).__name__}: {returned_value}")


class NodeActionNotDecoratedError(GraphOrchestratorException):
    def __init__(self, func):
        name = getattr(func, "__name__", repr(func))
        super().__init__(f"The function '{name}' passed to ProcessingNode must be decorated with @node_action.")


class RoutingFunctionNotDecoratedError(GraphOrchestratorException):
    def __init__(self, func):
        name = getattr(func, "__name__", repr(func))
        super().__init__(f"The function '{name}' passed to ConditionalEdge must be decorated with @routing_function.")


class InvalidAggregatorActionError(GraphOrchestratorException):
    def __init__(self, returned_value):
        super().__init__(f"Aggregator action must return a state, but got {type(returned_value).__name__}")


class AggregatorActionNotDecorated(GraphOrchestratorException):
    def __init__(self, func):
        name = getattr(func, "__name__", repr(func))
        super().__init__(f"The function '{name}' passed to Aggregator must be decorated with @aggregator_action")


class EmptyToolNodeDescriptionError(GraphOrchestratorException):
    def __init__(self, func):
        name = getattr(func, "__name__", repr(func))
        super().__init__(f"The tool function '{name}' has no description or docstring provided")


class ToolMethodNotDecorated(GraphOrchestratorException):
    def __init__(self, func):
        name = getattr(func, "__name__", repr(func))
        super().__init__(f"The function '{name}' passed to ToolNode has to be decorated with @tool_method")


@dataclass
class State:
    messages: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"State({self.messages}, metadata={self.metadata})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, State) and self.messages == other.messages and self.metadata == other.metadata

@dataclass
class RetryPolicy:
    max_retries: int = 3
    delay: float = 1.0
    backoff: float = 2.0

    def __str__(self) -> str:
        return f"RetryPolicy(max_retries={self.max_retries}, delay={self.delay:.2f}s, backoff={self.backoff:.2f}x)"

    def __repr__(self) -> str:
        return str(self)


def routing_function(func: Callable[[State], str]) -> Callable[[State], str]:
    @wraps(func)
    async def wrapper(state: State) -> str:
        result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
        if not isinstance(result, str):
            raise InvalidRoutingFunctionOutput(result)
        return result
    wrapper.is_routing_function = True
    return wrapper


def node_action(func: Callable[[State], State]) -> Callable[[State], State]:
    @wraps(func)
    async def wrapper(state: State) -> State:
        result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
        if not isinstance(result, State):
            raise InvalidNodeActionOutput(result)
        return result
    wrapper.is_node_action = True
    return wrapper


def tool_method(func: Callable[[State], State]) -> Callable[[State], State]:
    @wraps(func)
    async def wrapper(state: State) -> State:
        result = await func(state) if asyncio.iscoroutinefunction(func) else func(state)
        if not isinstance(result, State):
            raise InvalidToolMethodOutput(result)
        return result
    wrapper.is_node_action, wrapper.is_tool_method = True, True
    return wrapper


def aggregator_action(func: Callable[[List[State]], State]) -> Callable[[List[State]], State]:
    @wraps(func)
    async def wrapper(states: List[State]) -> State:
        result = await func(states) if asyncio.iscoroutinefunction(func) else func(states)  # type: ignore
        if not isinstance(result, State):
            raise InvalidAggregatorActionError(result)
        return result
    wrapper.is_aggregator_action = True
    return wrapper


@node_action
def passThrough(state: State) -> State:
    return state


@aggregator_action
def selectRandomState(states: List[State]) -> State:
    return random.choice(states)


class Node(ABC):
    def __init__(self, node_id: str) -> None:
        self.node_id, self.incoming_edges, self.outgoing_edges = node_id, [], []
        self.fallback_node_id: Optional[str] = None
        self.retry_policy: Optional[RetryPolicy] = None

    @abstractmethod
    def execute(self, state: State):
        raise NotImplementedError

    def set_fallback(self, fallback_node_id: str) -> None:
        self.fallback_node_id = fallback_node_id

    def set_retry_policy(self, retry_policy: RetryPolicy) -> None:
        self.retry_policy = retry_policy


class ProcessingNode(Node):
    def __init__(self, node_id: str, func: Callable[[State], State]) -> None:
        super().__init__(node_id)
        if not getattr(func, "is_node_action", False):
            raise NodeActionNotDecoratedError(func)
        self.func = func

    async def execute(self, state: State) -> State:
        return await self.func(state) if asyncio.iscoroutinefunction(self.func) else self.func(state)


class AggregatorNode(Node):
    def __init__(self, node_id: str, aggregator_action: Callable[[List[State]], State]) -> None:
        super().__init__(node_id)
        if not getattr(aggregator_action, "is_aggregator_action", False):
            raise AggregatorActionNotDecorated(aggregator_action)
        self.aggregator_action = aggregator_action

    async def execute(self, states: List[State]) -> State:
        return await self.aggregator_action(states) if asyncio.iscoroutinefunction(self.aggregator_action) else self.aggregator_action(states)


class ToolNode(ProcessingNode):
    def __init__(self, node_id: str, description: Optional[str], tool_method: Callable[[State], State]) -> None:
        if not getattr(tool_method, "is_tool_method", False):
            raise ToolMethodNotDecorated(tool_method)
        if not (description or (tool_method.__doc__ or "").strip()):
            raise EmptyToolNodeDescriptionError(tool_method)
        super().__init__(node_id, tool_method)
        self.description = description

    async def execute(self, state: State) -> State:
        return await self.func(state) if asyncio.iscoroutinefunction(self.func) else self.func(state)


class HumanInTheLoopNode(ProcessingNode):
    def __init__(self, node_id: str, interaction_handler: Callable[[State], Awaitable[State]], metadata: Optional[Dict[str, str]] = None) -> None:
        if not getattr(interaction_handler, "is_node_action", False):
            interaction_handler = node_action(interaction_handler)
        super().__init__(node_id, interaction_handler)
        self.metadata = metadata or {}

    async def execute(self, state: State) -> State:
        result = await self.func(state)
        if not isinstance(result, State):
            raise InvalidNodeActionOutput(result)
        return result

class AgentType(Enum):
    AzureOpenAI = "AzureOpenAI"

class IAgent(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def register_tool(self, name: str, func: Callable, schema: dict):
        pass

    @abstractmethod
    def run(self, state: State) -> State:
        pass

class AzureOpenAIAgent(IAgent):
    def __init__(self, config: dict):
        super().__init__(config)
        self.client = AzureOpenAI(
            azure_endpoint=config.get("azure_endpoint"),
            api_key=config.get("api_key"),
            api_version=config.get("api_version"),
        )
        self.deployment_name = config.get("deployment_name")
        self.max_steps = config.get("max_steps", 10)
        self.system_prompt = config.get(
            "system_prompt", "You are a helpful assistant."
        )

        # Tool registry
        self._tools: dict[str, Callable] = {}
        self._tool_schemas: list[dict] = []

    def register_tool(self, name: str, func: Callable, schema: dict):
        if not getattr(func, "is_tool_method", False):
            raise ToolMethodNotDecorated(func)
        self._tools[name] = func
        self._tool_schemas.append({
            "type": "function",
            "function": schema
        })

    async def run(self, state: State) -> State:
        steps = 0
        final_output: str | None = None

        messages = [
            {"role": "system", "content": self.system_prompt},
            state.messages[-1] if state.messages else {"role": "user", "content": ""}
        ]

        while steps < self.max_steps:
            steps += 1

            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=self._tool_schemas if self._tool_schemas else None,
                tool_choice="auto" if self._tool_schemas else None,
            )

            msg = response.choices[0].message
            messages.append(self.normalize_message(msg))
            self.print_conversation(messages)

            if not getattr(msg, "tool_calls", None):
                final_output = msg.content
                break

            async def call_tool(tool_call):
                fn = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")

                if fn not in self._tools:
                    return {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn,
                        "content": json.dumps({"error": f"Unknown tool {fn}"})
                    }

                func = self._tools[fn]

                try:
                    result_state = (
                        await func(State(messages=[{"role": "tool_input", "content": json.dumps(args)}]))
                        if asyncio.iscoroutinefunction(func)
                        else func(State(messages=[{"role": "tool_input", "content": json.dumps(args)}]))
                    )

                    result = result_state.messages[-1]["content"] if result_state.messages else str(result_state)
                except Exception as e:
                    result = {"error": str(e)}

                return {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": fn,
                    "content": json.dumps(result),
                }

            tool_results = await asyncio.gather(*(call_tool(tc) for tc in msg.tool_calls))
            messages.extend(tool_results)
            self.print_conversation(messages)

        if final_output:
            state.messages.append({"role": "assistant", "content": final_output})
        else:
            state.messages.append({
                "role": "assistant",
                "content": f"Aborted after {self.max_steps} steps"
            })

        return state
    
    @staticmethod
    def print_conversation(messages):
        print("\n=== Conversation so far ===")
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            tool_calls = m.get("tool_calls")
            name = m.get("name")

            if role == "tool":
                print(f"[TOOL → {name}] {content}")
            elif role == "assistant":
                if tool_calls:
                    print(f"[ASSISTANT] Requested tool call(s): {tool_calls}")
                else:
                    print(f"[ASSISTANT] {content}")
            else:
                print(f"[{role.upper()}] {content}")
        print("===========================\n")

    @staticmethod
    def normalize_message(m):
        if hasattr(m, "model_dump"):
            return m.model_dump()
        elif isinstance(m, dict):
            return m
        else:
            return dict(m)


class AgentFactory:
    @staticmethod
    def create(agent_type: AgentType, config: dict) -> Any:
        if agent_type == AgentType.AzureOpenAI:
            return AzureOpenAIAgent(config)
        raise ValueError(f"Unsupported agent type: {agent_type}")

class AgentNode(ProcessingNode):
    def __init__(self, node_id: str, agent_type: AgentType, config: str | dict) -> None:
        self.node_id = node_id
        self.agent_type = agent_type
        self.parsed_config = yaml.safe_load(config) if isinstance(config, str) else config
        self.agent = AgentFactory.create(agent_type, self.parsed_config)

        @node_action
        async def runner(state: State) -> State:
            if asyncio.iscoroutinefunction(self.agent.run):
                return await self.agent.run(state)
            else:
                return self.agent.run(state)
        super().__init__(node_id, runner)

    def register_tool(self, name: str, func: Callable, schema: dict):
        self.agent.register_tool(name, func, schema)


class Edge(ABC):
    pass


class ConcreteEdge(Edge):
    def __init__(self, source: Node, sink: Node):
        self.source, self.sink = source, sink


class ConditionalEdge(Edge):
    def __init__(self, source: Node, sinks: List[Node], router: Callable[[State], str]) -> None:
        if not getattr(router, "is_routing_function", False):
            raise RoutingFunctionNotDecoratedError(router)
        self.source, self.sinks, self.routing_function = source, sinks, router


class Graph:
    def __init__(self, start_node: Node, end_node: Node, name: Optional[str] = "graph") -> None:
        if not isinstance(start_node, Node) or not isinstance(end_node, Node):
            raise TypeError("start_node and end_node must be of type Node")
        self.nodes: Dict[str, Node] = {}
        self.concrete_edges: List[ConcreteEdge] = []
        self.conditional_edges: List[ConditionalEdge] = []
        self.start_node, self.end_node, self.name = start_node, end_node, name


class CheckpointData:
    def __init__(self, graph: Graph, initial_state: State, active_states: Dict[str, List[State]], superstep: int,
                 final_state: Optional[State], retry_policy: RetryPolicy, max_workers: int):
        self.graph, self.initial_state, self.active_states, self.superstep = graph, initial_state, active_states, superstep
        self.final_state, self.retry_policy, self.max_workers = final_state, retry_policy, max_workers

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "CheckpointData":
        with open(path, "rb") as f:
            return pickle.load(f)


class GraphBuilder:
    def __init__(self, name: Optional[str] = "graph"):
        start_node, end_node = ProcessingNode("start", passThrough), ProcessingNode("end", passThrough)
        self.graph = Graph(start_node, end_node, name)
        self.add_node(start_node)
        self.add_node(end_node)

    def add_node(self, node):
        if node.node_id in self.graph.nodes:
            raise DuplicateNodeError(node.node_id)
        self.graph.nodes[node.node_id] = node

    def set_fallback_node(self, node_id: str, fallback_node_id: str):
        if node_id not in self.graph.nodes:
            raise NodeNotFoundError(node_id)
        if fallback_node_id not in self.graph.nodes:
            raise NodeNotFoundError(fallback_node_id)
        self.graph.nodes[node_id].set_fallback(fallback_node_id)

    def set_node_retry_policy(self, node_id: str, retry_policy: RetryPolicy) -> None:
        if node_id not in self.graph.nodes:
            raise NodeNotFoundError(node_id)
        self.graph.nodes[node_id].set_retry_policy(retry_policy)

    def add_aggregator(self, aggregator: AggregatorNode):
        if aggregator.node_id in self.graph.nodes:
            raise DuplicateNodeError(aggregator.node_id)
        self.graph.nodes[aggregator.node_id] = aggregator

    def add_concrete_edge(self, source_id: str, sink_id: str):
        if source_id not in self.graph.nodes:
            raise NodeNotFoundError(source_id)
        if source_id == "end":
            raise GraphConfigurationError("End cannot be the source of a concrete edge")
        if sink_id not in self.graph.nodes:
            raise NodeNotFoundError(sink_id)
        if sink_id == "start":
            raise GraphConfigurationError("Start cannot be a sink of concrete edge")

        source, sink = self.graph.nodes[source_id], self.graph.nodes[sink_id]

        for edge in self.graph.concrete_edges:
            if edge.source == source and edge.sink == sink:
                raise EdgeExistsError(source_id, sink_id)

        for cond_edge in self.graph.conditional_edges:
            if cond_edge.source == source and sink in cond_edge.sinks:
                raise EdgeExistsError(source_id, sink_id)

        edge = ConcreteEdge(source, sink)
        self.graph.concrete_edges.append(edge)
        source.outgoing_edges.append(edge)
        sink.incoming_edges.append(edge)

    def add_conditional_edge(self, source_id: str, sink_ids: List[str], router: Callable[[State], str]):
        if source_id not in self.graph.nodes:
            raise NodeNotFoundError(source_id)
        if source_id == "end":
            raise GraphConfigurationError("End cannot be the source of a conditional edge")

        source, sinks = self.graph.nodes[source_id], []
        for sink_id in sink_ids:
            if sink_id not in self.graph.nodes:
                raise NodeNotFoundError(sink_id)
            if sink_id == "start":
                raise GraphConfigurationError("Start cannot be a sink of conditional edge")
            sinks.append(self.graph.nodes[sink_id])

        for e in self.graph.concrete_edges:
            if e.source == source and e.sink in sinks:
                raise EdgeExistsError(source_id, e.sink.node_id)

        for cond in self.graph.conditional_edges:
            if cond.source == source:
                for s in sinks:
                    if s in cond.sinks:
                        raise EdgeExistsError(source_id, s.node_id)

        edge = ConditionalEdge(source, sinks, router)
        self.graph.conditional_edges.append(edge)
        source.outgoing_edges.append(edge)
        for sink in sinks:
            sink.incoming_edges.append(edge)

    def build_graph(self) -> Graph:
        if any(isinstance(e, ConditionalEdge) for e in self.graph.start_node.outgoing_edges):
            raise GraphConfigurationError("Start node cannot have a conditional edge")
        if not any(isinstance(e, ConcreteEdge) for e in self.graph.start_node.outgoing_edges):
            raise GraphConfigurationError("Start node must have at least one outgoing concrete edge")
        if not self.graph.end_node.incoming_edges:
            raise GraphConfigurationError("End node must have at least one incoming edge")
        return self.graph


class GraphExecutor:
    def __init__(self, graph, initial_state, max_workers: int = 4, retry_policy: Optional[RetryPolicy] = None,
                 checkpoint_path: Optional[str] = None, checkpoint_every: Optional[int] = None,
                 allow_fallback_from_checkpoint: bool = False) -> None:
        self.graph, self.initial_state, self.max_workers = graph, initial_state, max_workers
        self.active_states: Dict[str, List[State]] = defaultdict(list)
        self.active_states[graph.start_node.node_id].append(initial_state)

        self.retry_policy = retry_policy if retry_policy else RetryPolicy(max_retries=0, delay=0)
        self.semaphore = asyncio.Semaphore(self.max_workers)
        self.checkpoint_path, self.checkpoint_every = checkpoint_path, checkpoint_every

        self.superstep, self.final_state = 0, None
        self.allow_fallback_from_checkpoint, self.already_retried_from_checkpoint = allow_fallback_from_checkpoint, False

        if self.allow_fallback_from_checkpoint and not self.checkpoint_path:
            raise GraphExecutionError("GraphExecutor", "Fallback from checkpoint is enabled, but no checkpoint_path is provided.")

    def to_checkpoint(self) -> CheckpointData:
        return CheckpointData(self.graph, self.initial_state, self.active_states, self.superstep,
                              self.final_state, self.retry_policy, self.max_workers)

    @classmethod
    def from_checkpoint(cls, chkpt: CheckpointData, checkpoint_path: Optional[str] = None, checkpoint_every: Optional[int] = None):
        executor = cls(chkpt.graph, chkpt.initial_state, chkpt.max_workers, chkpt.retry_policy, checkpoint_path, checkpoint_every)
        executor.active_states, executor.superstep, executor.final_state = chkpt.active_states, chkpt.superstep, chkpt.final_state
        return executor

    async def _execute_node_with_retry_async(self, node, input_data, retry_policy) -> None:
        retry_policy = node.retry_policy if node.retry_policy is not None else retry_policy
        attempt, delay = 0, retry_policy.delay

        while attempt <= retry_policy.max_retries:
            async with self.semaphore:
                try:
                    return await node.execute(input_data)
                except Exception as e:
                    if attempt == retry_policy.max_retries:
                        raise e
                    await asyncio.sleep(delay)
                    delay *= retry_policy.backoff
                    attempt += 1

    async def execute(self, max_supersteps: int = 100, superstep_timeout: float = 300.0) -> Optional[State]:
        final_state = None

        while self.active_states and self.superstep < max_supersteps:
            next_active_states, tasks = defaultdict(list), []

            for node_id, states in self.active_states.items():
                node = self.graph.nodes[node_id]
                input_data = states if isinstance(node, AggregatorNode) else copy.deepcopy(states[0])
                task = asyncio.create_task(asyncio.wait_for(self._execute_node_with_retry_async(node, input_data, self.retry_policy),
                                                            timeout=superstep_timeout))
                tasks.append((node_id, task, input_data))

            for node_id, task, original_input in tasks:
                node = self.graph.nodes[node_id]
                try:
                    result_state = await task

                except asyncio.TimeoutError:
                    if self.allow_fallback_from_checkpoint and not self.already_retried_from_checkpoint:
                        chkpt = CheckpointData.load(self.checkpoint_path)
                        fallback_executor = GraphExecutor.from_checkpoint(chkpt, checkpoint_path=self.checkpoint_path,
                                                                          checkpoint_every=self.checkpoint_every)
                        fallback_executor.allow_fallback_from_checkpoint, fallback_executor.already_retried_from_checkpoint = False, True
                        return await fallback_executor.execute(max_supersteps=max_supersteps, superstep_timeout=superstep_timeout)
                    raise GraphExecutionError(node_id, f"Execution timed out after {superstep_timeout}s.")

                except Exception as e:
                    fallback_id = getattr(node, "fallback_node_id", None)
                    if fallback_id:
                        fallback_node = self.graph.nodes[fallback_id]
                        try:
                            result_state = await asyncio.wait_for(self._execute_node_with_retry_async(fallback_node, original_input, self.retry_policy),
                                                                  timeout=superstep_timeout)
                        except Exception as fallback_error:
                            raise GraphExecutionError(fallback_id, f"Fallback node failed: {fallback_error}")
                    else:
                        raise GraphExecutionError(node_id, str(e))

                for edge in node.outgoing_edges:
                    if isinstance(edge, ConcreteEdge):
                        next_active_states[edge.sink.node_id].append(copy.deepcopy(result_state))
                    elif isinstance(edge, ConditionalEdge):
                        chosen_id = await edge.routing_function(result_state)
                        valid_ids = [sink.node_id for sink in edge.sinks]
                        if chosen_id not in valid_ids:
                            raise GraphExecutionError(node.node_id, f"Invalid routing output: '{chosen_id}'")
                        next_active_states[chosen_id].append(copy.deepcopy(result_state))

                if node_id == self.graph.end_node.node_id:
                    final_state = result_state

            self.active_states, self.superstep = next_active_states, self.superstep + 1

            if self.checkpoint_path and self.checkpoint_every and self.superstep % self.checkpoint_every == 0:
                self.to_checkpoint().save(self.checkpoint_path)

        if self.superstep >= max_supersteps:
            raise GraphExecutionError("N/A", "Max supersteps reached")
        return final_state