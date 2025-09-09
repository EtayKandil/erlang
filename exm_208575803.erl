-module(exm_YOUR_ID_HERE).

-export([
    exp_to_bdd/3,
    solve_bdd/2,
    listOfLeaves/1,
    reverseIteration/1,
    build_bdd_for_order/3 % Exported for use by the performance tester script
]).

-record(leaf_rec, {val, parents = []}).
-record(node_rec, {var, low, high, parents = []}).

%% ===================================================================
%% Public API Functions
%% ===================================================================

%%%-------------------------------------------------------------------
%%% @doc
%%% Converts a Boolean function to its optimal BDD representation based on
%%% the specified ordering criterion and data structure.
%%%
%%% `BoolFunc`: The Boolean function in tuple format.
%%% `Ordering`: Criterion for optimality (`tree_height`, `num_of_nodes`, `num_of_leafs`).
%%% `DS` (DataStructureType): `record` or `map`.
%%%
%%% This function explores all variable permutations to find the optimal BDD.
%%%-------------------------------------------------------------------
exp_to_bdd(BoolFunc, Ordering, DS) ->
    Start = erlang:monotonic_time(nanosecond),
    Vars  = get_vars_from_expr(BoolFunc),
    Perms = generate_permutations(Vars),
    BDDs  = [build_bdd_for_one_order(BoolFunc, P, DS) || P <- Perms],
    Result = pick_optimal_bdd(BDDs, Ordering),
    End = erlang:monotonic_time(nanosecond),
    TimeDiff = End - Start,
    io:format("exp_to_bdd execution time: ~p nanoseconds~n", [TimeDiff]),
    Result.

%%%-------------------------------------------------------------------
%%% @doc
%%% Solves a BDD for a given set of variable assignments.
%%%
%%% `BddTree`: The root of the BDD (record or map).
%%% `Assignments`: A list of `{VarName, Value}` tuples. Values can be
%%%                0/1 or true/false. Order of assignments does not matter.
%%%-------------------------------------------------------------------
solve_bdd(BddTree, Assignments) ->
    Start = erlang:monotonic_time(nanosecond),
    NormalizedAssignments = mapify_assignments(Assignments),
    Result = eval_bdd_tree(BddTree, NormalizedAssignments),
    End = erlang:monotonic_time(nanosecond),
    TimeDiff = End - Start,
    io:format("solve_bdd execution time: ~p nanoseconds~n", [TimeDiff]),
    Result.

%%%-------------------------------------------------------------------
%%% @doc
%%% Returns a list of pointers to all unique leaf nodes in the BDD tree.
%%%-------------------------------------------------------------------
listOfLeaves(BddTree) ->
    Start = erlang:monotonic_time(nanosecond),
    Result = collect_all_leaves(BddTree, []),
    End = erlang:monotonic_time(nanosecond),
    TimeDiff = End - Start,
    io:format("listOfLeaves execution time: ~p nanoseconds~n", [TimeDiff]),
    Result.

%%%-------------------------------------------------------------------
%%% @doc
%%% Given a leaf node, returns a human-readable list representing the
%%% shortest path from that leaf back to the root of the BDD.
%%% Path is [Leaf, Parent, ..., Root].
%%% Note: The PDF spec implies returning raw node structures. This version
%%% returns a formatted, readable path for user convenience as requested.
%%%-------------------------------------------------------------------
reverseIteration(Leaf) ->
    Start = erlang:monotonic_time(nanosecond),
    RawPath = find_path_to_root_bfs(Leaf),
    ReadablePathResult = format_path_for_reading(RawPath),
    End = erlang:monotonic_time(nanosecond),
    TimeDiff = End - Start,
    io:format("reverseIteration execution time: ~p nanoseconds~n", [TimeDiff]),
    ReadablePathResult.

%%%-------------------------------------------------------------------
%%% @doc
%%% Builds a BDD for a function given a specific variable order.
%%% Exported for use with the performance testing script.
%%%-------------------------------------------------------------------
build_bdd_for_one_order(Func, Order, record) ->
    {Root, _UniqueTable} = build_bdd_recursive(Func, Order, #{}, record),
    Root;
build_bdd_for_one_order(Func, Order, map) ->
    {Root, _UniqueTable} = build_bdd_recursive(Func, Order, #{}, map),
    Root.

%% ===================================================================
%% Helpers for exp_to_bdd/3
%% ===================================================================

%% Extracts unique variables from a Boolean function expression.
get_vars_from_expr(Expr) ->
    lists:usort(do_get_vars(Expr, [])).

do_get_vars(Var, Acc) when is_atom(Var), Var =/= true, Var =/= false -> % Vars are atoms
    case lists:member(Var, Acc) of
        true  -> Acc;
        false -> [Var | Acc]
    end;
do_get_vars({'not', A}, Acc) ->
    do_get_vars(A, Acc);
do_get_vars({Operator, {A, B}}, Acc) when Operator == 'and'; Operator == 'or' ->
    do_get_vars(A, do_get_vars(B, Acc));
do_get_vars(_Other, Acc) -> % Handles constants or already processed parts
    Acc.

%% Generates all permutations of a list.
generate_permutations([]) -> [[]];
generate_permutations(L)  -> [[X|P] || X <- L, P <- generate_permutations(L--[X])].

%% Selects the optimal BDD from a list based on a criterion.
pick_optimal_bdd([FirstBDD | RestBDDs], Criterion) ->
    pick_optimal_bdd_recursive(RestBDDs, FirstBDD, Criterion).

pick_optimal_bdd_recursive([], BestBDDSoFar, _Criterion) ->
    BestBDDSoFar;
pick_optimal_bdd_recursive([CurrentBDD | TailBDDs], BestBDDSoFar, tree_height) ->
    BetterBDD = case get_bdd_height(CurrentBDD) < get_bdd_height(BestBDDSoFar) of
        true  -> CurrentBDD;
        false -> BestBDDSoFar
    end,
    pick_optimal_bdd_recursive(TailBDDs, BetterBDD, tree_height);
pick_optimal_bdd_recursive([CurrentBDD | TailBDDs], BestBDDSoFar, num_of_nodes) ->
    BetterBDD = case get_bdd_node_count(CurrentBDD) < get_bdd_node_count(BestBDDSoFar) of
        true  -> CurrentBDD;
        false -> BestBDDSoFar
    end,
    pick_optimal_bdd_recursive(TailBDDs, BetterBDD, num_of_nodes);
pick_optimal_bdd_recursive([CurrentBDD | TailBDDs], BestBDDSoFar, num_of_leafs) ->
    BetterBDD = case get_bdd_leaf_count(CurrentBDD) < get_bdd_leaf_count(BestBDDSoFar) of
        true  -> CurrentBDD;
        false -> BestBDDSoFar
    end,
    pick_optimal_bdd_recursive(TailBDDs, BetterBDD, num_of_leafs).

%% ===================================================================
%% Core BDD Construction Logic
%% ===================================================================

%%% @doc
%%% Recursively builds a BDD based on Shannon expansion.
%%% `Expr`: Current Boolean expression (or substituted value).
%%% `Order`: List of remaining variables to expand on.
%%% `UniqueTable`: Map used to store and retrieve unique BDD nodes (ensures sharing).
%%% `DS`: Data structure type (`record` or `map`).
%%% Returns `{Node, UpdatedUniqueTable}`.
build_bdd_recursive(Expr, [], UniqueTable, DS) -> % Base case: no more variables
    TerminalValue = eval_constant_expr(Expr),
    create_leaf_node(TerminalValue, DS, UniqueTable);
build_bdd_recursive(Expr, [CurrentVar | RemainingVars], UniqueTableIn, DS) ->
    % Shannon expansion: f = not(CurrentVar)*f_low + CurrentVar*f_high

    % Cofactor for CurrentVar = 0 (low branch)
    ExprLow = substitute_var(Expr, CurrentVar, 0),
    {LowChild, UniqueTableMid} = build_bdd_recursive(ExprLow, RemainingVars, UniqueTableIn, DS),

    % Cofactor for CurrentVar = 1 (high branch)
    ExprHigh = substitute_var(Expr, CurrentVar, 1),
    {HighChild, UniqueTableOut} = build_bdd_recursive(ExprHigh, RemainingVars, UniqueTableMid, DS),

    % Create internal node (or reuse if identical children or existing node)
    create_internal_node(CurrentVar, LowChild, HighChild, DS, UniqueTableOut).

%% Creates a leaf node, ensuring uniqueness via UniqueTable.
create_leaf_node(Value, record, UniqueTable) ->
    Key = {record, leaf, Value}, % Canonical key for this leaf
    case maps:get(Key, UniqueTable, undefined) of
        undefined -> % New leaf
            Leaf = #leaf_rec{val = Value, parents = []},
            {Leaf, maps:put(Key, Leaf, UniqueTable)};
        ExistingLeaf -> % Existing leaf
            {ExistingLeaf, UniqueTable}
    end;
create_leaf_node(Value, map, UniqueTable) ->
    Key = {map, leaf, Value}, % Canonical key for this leaf
    case maps:get(Key, UniqueTable, undefined) of
        undefined -> % New leaf
            Leaf = #{type => leaf, val => Value, parents => []},
            {Leaf, maps:put(Key, Leaf, UniqueTable)};
        ExistingLeaf -> % Existing leaf
            {ExistingLeaf, UniqueTable}
    end.

%% Creates an internal node, applying reduction rules and ensuring uniqueness.
create_internal_node(Var, LowChild, HighChild, DS, UniqueTable) ->
    % Reduction Rule: If low and high children are identical, return the child.
    if LowChild == HighChild ->
        {LowChild, UniqueTable};
       true ->
        do_create_internal_node(Var, LowChild, HighChild, DS, UniqueTable)
    end.

%% Actual node creation after checking for identical children rule.
%% Ensures sharing of isomorphic nodes via UniqueTable.
do_create_internal_node(Var, LowChild, HighChild, record, UniqueTable) ->
    % Canonical key for this node structure
    Key = {record, node, Var, LowChild, HighChild},
    case maps:get(Key, UniqueTable, undefined) of
        undefined -> % New node
            Node = #node_rec{var = Var, low = LowChild, high = HighChild, parents = []},
            % Add this node as a parent to its children
            {UpdatedLowChild, _} = add_parent_to_child(LowChild, Node, record, UniqueTable),
            {UpdatedHighChild, _} = add_parent_to_child(HighChild, Node, record, UniqueTable),
            % Create the node with potentially updated children references (though add_parent_to_child returns original UniqueTable)
            FinalNode = Node#node_rec{low = UpdatedLowChild, high = UpdatedHighChild},
            {FinalNode, maps:put(Key, FinalNode, UniqueTable)};
        ExistingNode -> % Existing isomorphic node
            {ExistingNode, UniqueTable}
    end;
do_create_internal_node(Var, LowChild, HighChild, map, UniqueTable) ->
    Key = {map, node, Var, LowChild, HighChild},
    case maps:get(Key, UniqueTable, undefined) of
        undefined -> % New node
            Node = #{type => node, var => Var, low => LowChild, high => HighChild, parents => []},
            {UpdatedLowChild, _} = add_parent_to_child(LowChild, Node, map, UniqueTable),
            {UpdatedHighChild, _} = add_parent_to_child(HighChild, Node, map, UniqueTable),
            FinalNode = Node#{low => UpdatedLowChild, high => UpdatedHighChild},
            {FinalNode, maps:put(Key, FinalNode, UniqueTable)};
        ExistingNode -> % Existing isomorphic node
            {ExistingNode, UniqueTable}
    end.

%% Adds ParentNode to the parents list of ChildNode.
%% Returns {UpdatedChildNode, UnchangedUniqueTable} - UniqueTable is passed for interface consistency
%% but not modified here as child modification does not change its identity key in the table.
add_parent_to_child(ChildNode, ParentNode, record, UniqueTable) when is_record(ChildNode, leaf_rec) ->
    UpdatedChild = ChildNode#leaf_rec{parents = [ParentNode | ChildNode#leaf_rec.parents]},
    {UpdatedChild, UniqueTable};
add_parent_to_child(ChildNode, ParentNode, record, UniqueTable) when is_record(ChildNode, node_rec) ->
    UpdatedChild = ChildNode#node_rec{parents = [ParentNode | ChildNode#node_rec.parents]},
    {UpdatedChild, UniqueTable};
add_parent_to_child(ChildMap, ParentNode, map, UniqueTable) when is_map(ChildMap) ->
    CurrentParents = maps:get(parents, ChildMap, []),
    UpdatedChild = ChildMap#{parents := [ParentNode | CurrentParents]},
    {UpdatedChild, UniqueTable}.

%% Substitutes a variable in an expression with a constant value (0 or 1).
substitute_var(VarAtom, VarAtom, Value) when is_atom(VarAtom) -> Value; % Matched variable
substitute_var(OtherAtom, _VarToSub, _Value) when is_atom(OtherAtom) -> OtherAtom; % Different variable or true/false
substitute_var(Constant, _, _) when is_integer(Constant) -> Constant; % 0 or 1
substitute_var({'not', Arg}, VarToSub, Value) ->
    SubArg = substitute_var(Arg, VarToSub, Value),
    if is_integer(SubArg) -> 1 - SubArg; % Optimization: Evaluate if constant
       true -> {'not', SubArg}
    end;
substitute_var({Operator, {Arg1, Arg2}}, VarToSub, Value) when Operator == 'and'; Operator == 'or' ->
    SubArg1 = substitute_var(Arg1, VarToSub, Value),
    SubArg2 = substitute_var(Arg2, VarToSub, Value),
    if is_integer(SubArg1), is_integer(SubArg2) -> % Optimization: Evaluate if both are constant
        if Operator == 'and' -> SubArg1 band SubArg2;
           Operator == 'or'  -> SubArg1 bor SubArg2
        end;
       true -> {Operator, {SubArg1, SubArg2}}
    end;
substitute_var(UnchangedExpr, _VarToSub, _Value) -> UnchangedExpr. % Should not be reached for well-formed expressions

%% Evaluates a Boolean expression assumed to contain only constants (0 or 1).
eval_constant_expr(0) -> 0;
eval_constant_expr(1) -> 1;
eval_constant_expr(true) -> 1; % Handle boolean atoms
eval_constant_expr(false) -> 0;
eval_constant_expr({'not', Arg}) ->
    1 - eval_constant_expr(Arg);
eval_constant_expr({'and', {Arg1, Arg2}}) ->
    eval_constant_expr(Arg1) band eval_constant_expr(Arg2);
eval_constant_expr({'or', {Arg1, Arg2}}) ->
    eval_constant_expr(Arg1) bor eval_constant_expr(Arg2).

%% ===================================================================
%% Helpers for solve_bdd/2
%% ===================================================================

%% Converts a list of variable assignments to a map for quick lookup.
%% Normalizes true/false to 1/0.
mapify_assignments(AssignmentsList) ->
    maps:from_list([{Var, normalize_boolean_value(Val)} || {Var, Val} <- AssignmentsList]).

normalize_boolean_value(true) -> 1;
normalize_boolean_value(false) -> 0;
normalize_boolean_value(1) -> 1;
normalize_boolean_value(0) -> 0.

%% Traverses the BDD according to variable assignments to find the result.
eval_bdd_tree(#leaf_rec{val = Value}, _Assignments) -> Value;
eval_bdd_tree(#node_rec{var = VarName, low = LowChild, high = HighChild}, Assignments) ->
    case maps:get(VarName, Assignments) of
        0 -> eval_bdd_tree(LowChild, Assignments);
        1 -> eval_bdd_tree(HighChild, Assignments)
    end;
eval_bdd_tree(MapNode = #{type := leaf}, _Assignments) ->
    maps:get(val, MapNode);
eval_bdd_tree(MapNode = #{type := node}, Assignments) ->
    VarName = maps:get(var, MapNode),
    case maps:get(VarName, Assignments) of
        0 -> eval_bdd_tree(maps:get(low, MapNode), Assignments);
        1 -> eval_bdd_tree(maps:get(high, MapNode), Assignments)
    end.

%% ===================================================================
%% Helpers for listOfLeaves/1
%% ===================================================================

%% Recursively collects all unique leaf nodes.
collect_all_leaves(LeafNode = #leaf_rec{}, Acc) ->
    if lists:member(LeafNode, Acc) -> Acc; true -> [LeafNode | Acc] end;
collect_all_leaves(Node = #node_rec{}, Acc) ->
    Acc1 = collect_all_leaves(Node#node_rec.low, Acc),
    collect_all_leaves(Node#node_rec.high, Acc1);
collect_all_leaves(LeafMap = #{type := leaf}, Acc) ->
    if lists:member(LeafMap, Acc) -> Acc; true -> [LeafMap | Acc] end;
collect_all_leaves(NodeMap = #{type := node}, Acc) ->
    Acc1 = collect_all_leaves(maps:get(low, NodeMap), Acc),
    collect_all_leaves(maps:get(high, NodeMap), Acc1).

%% ===================================================================
%% Helpers for reverseIteration/1
%% ===================================================================

%% Performs a Breadth-First Search from a leaf upwards using parent pointers.
%% VisitedSet tracks nodes already processed to handle shared parents and avoid loops.
find_path_to_root_bfs(StartLeaf) ->
    do_find_path_bfs([{StartLeaf, [StartLeaf]}], sets:new()). % Queue stores {Node, PathToNode}

do_find_path_bfs([{#leaf_rec{parents=[]}, PathSoFar} | _QueueRest], _VisitedSet) -> lists:reverse(PathSoFar);
do_find_path_bfs([{#node_rec{parents=[]}, PathSoFar} | _QueueRest], _VisitedSet) -> lists:reverse(PathSoFar);
do_find_path_bfs([{MapNode = #{type := leaf, parents := []}, PathSoFar} | _QueueRest], _VisitedSet) -> lists:reverse(PathSoFar);
do_find_path_bfs([{MapNode = #{type := node, parents := []}, PathSoFar} | _QueueRest], _VisitedSet) -> lists:reverse(PathSoFar);
do_find_path_bfs([{CurrentNode, PathSoFar} | QueueRest], VisitedSet) ->
    case sets:is_element(CurrentNode, VisitedSet) of
        true  -> do_find_path_bfs(QueueRest, VisitedSet); % Already processed this node via a (possibly shorter) path
        false ->
            Parents = get_node_parents(CurrentNode),
            NewVisitedSet = sets:add_element(CurrentNode, VisitedSet),
            NewQueueEntries = [{Parent, [Parent | PathSoFar]} || Parent <- Parents],
            do_find_path_bfs(QueueRest ++ NewQueueEntries, NewVisitedSet)
    end;
do_find_path_bfs([], _VisitedSet) -> % Should not happen if graph is connected to a root with no parents
    []; % Or error: {error, path_not_found_to_root_without_parents}
    % Given BDD structure, a root should always be reached.
    % This clause handles empty queue if path somehow doesn't terminate at a root.
    % If StartLeaf itself has no parents, the initial clauses handle it.
    [].


%% Gets the list of parent nodes for a given BDD node.
get_node_parents(#leaf_rec{parents = P}) -> P;
get_node_parents(#node_rec{parents = P}) -> P;
get_node_parents(MapNode) when is_map(MapNode) -> maps:get(parents, MapNode, []).

%% Formats a raw path (list of BDD nodes) into a readable list of tuples.
format_path_for_reading(PathList) ->
    lists:map(fun(Node) ->
                  if
                      is_record(Node, leaf_rec) ->
                          {leaf, Node#leaf_rec.val};
                      is_record(Node, node_rec) ->
                          {node, Node#node_rec.var};
                      is_map(Node) ->
                          case maps:get(type, Node, undefined) of
                              leaf -> {leaf, maps:get(val, Node)};
                              node -> {node, maps:get(var, Node)};
                              _    -> {unknown_map_node_in_path, Node}
                          end;
                      true ->
                          {unknown_node_type_in_path, Node}
                  end
              end, PathList).

%% ===================================================================
%% BDD Metric Calculation Helpers (used by pick_optimal_bdd)
%% These use sets/maps to correctly count in DAGs with shared nodes.
%% ===================================================================

get_bdd_height(RootNode) ->
    {H, _Memo} = do_get_bdd_height(RootNode, maps:new()), H.

do_get_bdd_height(Node, Memo) when is_record(Node, leaf_rec); (is_map(Node) andalso maps:get(type, Node, undefined) == leaf) ->
    {0, Memo}; % Height of a leaf is 0
do_get_bdd_height(Node, Memo) ->
    case maps:is_key(Node, Memo) of % Memoization using node itself as key
        true  -> {maps:get(Node, Memo), Memo};
        false ->
            LowChild  = if is_record(Node, node_rec) -> Node#node_rec.low;  true -> maps:get(low, Node) end,
            HighChild = if is_record(Node, node_rec) -> Node#node_rec.high; true -> maps:get(high, Node) end,
            {HeightLow, Memo1} = do_get_bdd_height(LowChild, Memo),
            {HeightHigh, Memo2} = do_get_bdd_height(HighChild, Memo1),
            NodeHeight = 1 + erlang:max(HeightLow, HeightHigh),
            {NodeHeight, maps:put(Node, NodeHeight, Memo2)}
    end.

get_bdd_node_count(RootNode) ->
    {N, _Visited} = do_get_bdd_node_count(RootNode, sets:new()), N.

do_get_bdd_node_count(Node, Visited) when is_record(Node, leaf_rec); (is_map(Node) andalso maps:get(type, Node, undefined) == leaf) ->
    {0, Visited}; % Leaves are not counted as internal nodes
do_get_bdd_node_count(Node, Visited) ->
    case sets:is_element(Node, Visited) of
        true  -> {0, Visited}; % Already counted
        false ->
            Visited1 = sets:add_element(Node, Visited),
            LowChild  = if is_record(Node, node_rec) -> Node#node_rec.low;  true -> maps:get(low, Node) end,
            HighChild = if is_record(Node, node_rec) -> Node#node_rec.high; true -> maps:get(high, Node) end,
            {CountLow, VisitedL} = do_get_bdd_node_count(LowChild, Visited1),
            {CountHigh, VisitedH} = do_get_bdd_node_count(HighChild, VisitedL),
            {1 + CountLow + CountHigh, VisitedH}
    end.

get_bdd_leaf_count(RootNode) ->
    {N, _Visited} = do_get_bdd_leaf_count(RootNode, sets:new()), N.

do_get_bdd_leaf_count(Node, Visited) when is_record(Node, leaf_rec); (is_map(Node) andalso maps:get(type, Node, undefined) == leaf) ->
    case sets:is_element(Node, Visited) of % Check if this unique leaf instance was counted
        true  -> {0, Visited};
        false -> {1, sets:add_element(Node, Visited)}
    end;
do_get_bdd_leaf_count(Node, Visited) -> % Internal node
    case sets:is_element(Node, Visited) of
        true  -> {0, Visited}; % Path through this node already explored for leaves
        false ->
            Visited1 = sets:add_element(Node, Visited),
            LowChild  = if is_record(Node, node_rec) -> Node#node_rec.low;  true -> maps:get(low, Node) end,
            HighChild = if is_record(Node, node_rec) -> Node#node_rec.high; true -> maps:get(high, Node) end,
            {CountLow, VisitedL} = do_get_bdd_leaf_count(LowChild, Visited1),
            {CountHigh, VisitedH} = do_get_bdd_leaf_count(HighChild, VisitedL),
            {CountLow + CountHigh, VisitedH} % Sum leaves from children
    end.