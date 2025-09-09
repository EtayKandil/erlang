-module(exm_208575803_test).

-export([
    test_complex_bdd/0
]).

test_complex_bdd() ->
    Expr = 
        {'or',
            { {'and', {x1, {'or', {x2, {'not', x3}}}}},
              {'and', {x4, {'not', x2}}} }
        },

    Assignments = [
        [{x1,0},{x2,0},{x3,0},{x4,0}], % 0
        [{x1,0},{x2,0},{x3,0},{x4,1}], % 1
        [{x1,0},{x2,0},{x3,1},{x4,0}], % 0
        [{x1,0},{x2,0},{x3,1},{x4,1}], % 1
        [{x1,0},{x2,1},{x3,0},{x4,0}], % 1
        [{x1,0},{x2,1},{x3,0},{x4,1}], % 1
        [{x1,0},{x2,1},{x3,1},{x4,0}], % 0
        [{x1,0},{x2,1},{x3,1},{x4,1}], % 0
        [{x1,1},{x2,0},{x3,0},{x4,0}], % 1
        [{x1,1},{x2,0},{x3,0},{x4,1}], % 1
        [{x1,1},{x2,0},{x3,1},{x4,0}], % 1
        [{x1,1},{x2,0},{x3,1},{x4,1}], % 1
        [{x1,1},{x2,1},{x3,0},{x4,0}], % 1
        [{x1,1},{x2,1},{x3,0},{x4,1}], % 1
        [{x1,1},{x2,1},{x3,1},{x4,0}], % 1
        [{x1,1},{x2,1},{x3,1},{x4,1}]  % 1
    ],

    Expected = [0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1],

    RootRecord = exm_208575803:exp_to_bdd(Expr, tree_height, record),
    RootMap    = exm_208575803:exp_to_bdd(Expr, tree_height, map),

    io:format("==== Testing RECORD BDD ====\n"),
    lists:foreach(
        fun({Assign, ExpVal}) ->
            Shuffled = shuffle(Assign),
            Real = exm_208575803:solve_bdd(RootRecord, Shuffled),
            case Real == ExpVal of
                true  -> io:format("[PASS] Assign=~p => Got=~p Expected=~p~n", [Shuffled, Real, ExpVal]);
                false -> io:format("[FAIL] Assign=~p => Got=~p Expected=~p~n", [Shuffled, Real, ExpVal])
            end
        end,
        lists:zip(Assignments, Expected)
    ),

    io:format("==== Testing MAP BDD ====\n"),
    lists:foreach(
        fun({Assign, ExpVal}) ->
            Shuffled = shuffle(Assign),
            Real = exm_208575803:solve_bdd(RootMap, Shuffled),
            case Real == ExpVal of
                true  -> io:format("[PASS] Assign=~p => Got=~p Expected=~p~n", [Shuffled, Real, ExpVal]);
                false -> io:format("[FAIL] Assign=~p => Got=~p Expected=~p~n", [Shuffled, Real, ExpVal])
            end
        end,
        lists:zip(Assignments, Expected)
    ).

%% Small helper: shuffle a list randomly
shuffle(List) ->
    RandomTagged = [{rand:uniform(1000), X} || X <- List],
    Sorted = lists:sort(fun({A,_}, {B,_}) -> A =< B end, RandomTagged),
    [X || {_, X} <- Sorted].
