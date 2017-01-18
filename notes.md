# ETH ALGORITHM LAB 2016 FALL



## Dynamic Programming

__Reference: Slide of Stanford CS 97SI__

> 1. Three steps: 
>
>    1. define __subproblem__ [Important]
>
>    2. find the __recurrence__ that relates subproblems
>
>    3. solve the __base cases__
>
>       DP solve (overlapped) subproblems only once and save the solutions of these subproblems for later usage.

```c++
1-dimension DP Example
Problem: given n, find the number of different ways to write n as the sum of 1, 3, 4

Subproblem: D_n be the number of ways to write n as the sum of 1, 3, 4
recurrence relation: D_n = D_{n-1} + D_{n-3} + D_{n-4}
```

```
2-dimension DP Example
Problem: given two strings x and y, find the longest common subsequence (LCS)

Subproblem: Let D_{ij} be the length of the LCS of x_{1,...,i} and y_{1,...,j}
```

```
Interval DP Example
Problem: given a string x = x_{1,...,n}, find the minimum number of characters that need to be inserted to make it a palindrome

Subproblem: 
```

```
Tree DP Example
Problem: given a tree, color nodes black as many as possible without coloring two adjacent nodes

Subproblem: for root node r -->  B_r: the optimal solution for a subtree having r as the root, where r is colored; W_r: the optimal solution for a subtree haveing r as the root, where r is not colored.
```

```
Subset DP Example
Problem: given a weighted graph with n nodes, find the shortest path that visits every node exactly once

Subproblem: D_{S,v}: the length of optimal path that visits every node in the set S exactly once and end in v.
Recurrence: D_{S,v} = min_{u} (D_{S-{v},u} + cost(u,v))
```



#### Problems:

* _Burning Coins from Two Sides(week 5)_: Interval DP. 

  * f(i,j) := max award from i to j

  * if(j-i) >= 2: f(i,j) = max{(V[i] + min{f(i+2, j), f(i+i, j-1)} ), (V[j] + min{f(i+1, j-1), f(i, j-2)} ) }

    if(j-i) == 1: f(i,j) = max{V[i], V[j]}

    if(j-i) == 0: f(i,j) = V[i]

* _Light Pattern(week 5)_: 1-dimension DP

  * f_0(i) := minimum operations to make first i groups same with given pattern
  * f_1(i) := minimum operations to make first i groups totally different with given pattern

* _The Great Game(week 5)_: 1-dimension DP

  * M0(i) := shortest steps from i'th transition entry to destination
  * M1(i) := longest steps from i'th transition engty to destination

* _Punch(week 11)_: 2-dimension DP

* _Poker Chips(week 5)_: 5-dimension DP. Complexity: complete the state matrix requires DOT(mi + 1) <= 2^16; and in order to complete each element in the state matrix, we need iterate 2^5 different choices and find the optmal one.



## Binary Search

> Goal: find the smallest k that is ‘large enough’.
>
> Properties: (1) for a fixed k, you can check __efficiently__ if it is ‘large enough’, (2) The property of not being too small is __monotone__. [Slide 2]

```c++
Binary Search Outline
int lmin = 0, lmax = 1;
while (too_small(lmax)) lmax *= 2;
while (lmin != lmax){
  int p = (lmin+lmax)/2;
  if (too_small(p))
    lmin = p+1;
  else
    lmax = p;
}
L = lmin;
```

#### problems:

* _Evoluation(week2)_
* _TheeV(week 4)_: sorted cities with ascending distance to the first TV transmitter. Binary search the position with splits cities into two parts.



## Sliding Window / Linear Iteration

> Iterate all elements and update the `candidate_state_variable` on the fly. Return the optimal `state_variable`.

> Two key components: (1) a state variable `state_variable` (e.g. the given number in _Deck of Cards_ and the fartest position in _Dominoes_) and (2) a current(interesting) interval `[l, r]`.  We slide the inteval from left to right and update the `state_variable`.  __IMPORTANT: what is the relationship between `state_variable` and sliding window__   Finally, return the optimal `state_variable`

> Special case, only one element in the sliding window (Iterate all elements). E.g. _Deck of Cards_ 

> The problem should be solved in linear time. Maybe need sort at beginning. 

#### Problems:

* _Dominoes (week1)_  Variable `pos`: the farthest position of domino that would fall down. Iterator each domino check (1) whether this dominon would fall down and (2) if this domino falls down, update `pos`
* _Deck of Cards(week 1)_ For the given `k`, maintain a state variable `current_sum` for candidate and a sliding window. Update `current_sum` when current window changes.
* _Attack of the clones(week 3)_  Variable `count_Jedi_number`: the number of Jedi that protect this position. Each element is either a `start_of_Jedi` or `end_of_Jedi`. Iterate each element, `count_Jedi_number++` if meet a `start_of_Jedi` or  `count_Jedi_number--` if meet a `end_of_Jedi`. 
* _Search Snippets(week 2)_ 
* _Moving books(week 2)_: Sort books and people by its weight and strength. Iterate each book and allocate it to suitable people.
* _Octopussy(week2)_: Iterate each (leaves) balls one-by-one. Use a priority queue for help.
* _Hiking maps(week 3)_
* _Attack of the Clones(week 3)_: Iterate beginning points and ending points to find the segment protected by less 10 Jedi.



## Greedy Method

> In Slide 2.

> Greedy algorithm __could be used__ when locally optimal choices result in global __optimum__.

> Common steps:
>
> 1. Modeling: task requires you to __construct a particular set__ that is in some sense globally optimal.
> 2. Greedy choice: given already chosen elements c1, …, ck-1, decide __how choose ck__ , based on some local optimality criterion.
> 3. Prove that elements obtained in this way __result in a globally optimal set__. (e.g. __exchange argument__)
> 4. Implement the algorithm efficiently.

> Example: MST —> find the edge set that forms spanning tree with minimum weight; Interval scheduling —> find the maximum set of compatible jobs. (__Sorting__ is a very useful technique) 



#### problems:

* _Boats(week 2)_:  Ground set: boats. Greedy choice: choose the next boat whose end point is minimal.
* _Attack of the Clones(week 3)_

## Brute Force / Split and List

> Example problem: Is there a subset of S which sums to k? (if k is small, it could be solved by DP withO(n*k) ). 
>
> Search all possible strategies: in each strategy, we make a decision for each item that whether or not to take this item. ==> n is small O(n*2^(n/2)), n<=40.
>
> Methods: __Backtracking__ V.S. __Representing set as bits__ [Slide 5]



#### Problems:

* _Light at the Museum(week 5)_: Similar to the example problem, but need to check __M constraints__. 

## CGAL - Intro

> Combinatorial Algorithm <— Geometric Predicate <— Algebraic Computation
>
> 1.  algebraic computation is non-trivial 
> 2. Filtering to ganrantee correctness: Check whether things go fine and use exact algebra only 		when needed
> 3. Exact construction is very expensive ==> If possible, use predicates with numerical tests decide branching instead of exact construction

```
double could store 53-bit integer(without +-*/); and 25-bit integer for + - *
32bit integer reads as int, 64bit integer reads as long
```

> (1) K::Point_2; K::FT; std::setiosflags(std::ios::fixed); std::setprecision(2); point.x(); point.y(); (2) return type of intersection; (3) bounding volumes [slide 3]

``` bash
CGAL usage
cgal_create_cmake_script
cmake .
make
```



#### problems:

* _Hit(week 3)_: use _long_ to save the 51-bits integers
* _First hit(week3)_: try to avoid constructions and use ```std::random_shuffle()``` 
* _Hiking maps(week 3)_: together with sliding window.
* _Antenna(week 3)_: minimum enclosing circle
* _TheeV(week 4)_: call minimum enclosing circle

## LP/QP



## Delaunay Triangulation



## BGL

> 1. __Graph Traverse: BFS, DFS__
>
>    Two considerations: (1) Nodes in graph would be seen as 'abstract states' —> answer the questions like:  whether some state could reached from the given state (e.g. _odd route_), or what is the shortest path from one state to another state. (2) We are constructing path __incrementally__, which suggests we could answer queries __on the fly__.

```c++
BFS implementation
queue<int> q;
vector<int> pushed(n, 0);
q.push_back(v0);
pushed[v0] = 1;
while (not q.empty()) {
	int v = q.front();
	// do something for v
	q.pop_front();
	for (int u : adj[v]) {
		if (not pushed[u]) {
		q.push_back(u);
		pushed[u] = 1;
		}
	}
}
```

```c++
DFS implementation
vector<int> visited(n, 0);
void dfs(int v) {
	// do something for v
	for (int u : adj[v]) {
		if (not visited[u]) {
		visited[u] = 1;
		dfs(u);
		}
	}
	// maybe do something else for v
}
```

#### problems:

* _Evolution(week 2)_
* _Odd Route(week 10)_: what are the nodes (abstract states)?



> 2. Basic Graph Algorithms
>    - Distance: Dijkstra shortest paths, Prim minimum spanning tree, Kruskal minimum spanning tree.
>    - Components: Connected, biconnected, strongly connected components
>    - Matchings: General unweighted matching
>    - Others: topological sorting; Eulerian tours; Union-Find (e.g. _GoldenEye_).
> 3. Dijkstra shortest path __complexity__: ```O(V*logV + E)```, Dijkstra's algorithm finds all the shortest paths from the __source vertex__ to __every other vertex__.
> 4. Strongly connected components complexity: ```O(v+E)```. 
> 5. Biconnected subgraph: the biconnected components of a graph are the maximal subsets of vertices such that the __removal of a vertex__ from a particular component __will not__ disconnect the component. Complexity ```O(V+E)```
> 6. Prim MST: add the closest undiscovered neighbor of all discovered neighbors; Kruskal MST: add the next shortest edge without creating a cycle 
> 7. BGL __does not__ provide weighted matching algorithms.

```c++
Union-Find in C++
#include <boost/pending/disjoint_sets.hpp>
typedef boost::disjoint_sets_with_storage<> Uf;
Uf ufa(num_elements);
ufa.find_set(i) // return the representation of i
ufa.union_set(i, j) // merge 
```

#### problems:

* _GoldenEye(week 12)_: 
* _Ant Challenge(week 4)_: 
* _Important Bridges(week 4)_: biconnected connected graph
* _Buddy Selection(week 4)_: maximum matching on unweighted graph.



## Max Flow / Max Flow Min Cost

> The MaxFlow (MinCost) problem
>
> - some quantities are reserved in some sources. These quantities want to go to the sinks.
> - edge (u, v): the quantity can go from u (abstract state) to v (abstract state)
> - different decisions <==> different path for the quantities. In more details, in most cases, we have decision variables and the objective function. What we want to do is to choose the particular decision that make objective function optimal. When we change decisions, we change the network structure for the flow (e.g. _On Her Majesty's Secret Service_) or change the capacity of edges. This results in different flow!
> - final distribution of quantities &/ distribution of flow is related with the objective function
> - max flow algorithm  => choose the routes that maximize flow (min cost) ==> this should correspond to the optimal decision.

> Common tricks:
>
> 1. With edge capacity set to 1, the maximum number of edge-disjoint s-t-paths is equal to the maximum flow from s to t.
> 2. Max flow with lower bound on edge: .

#### Problems:

* _On her Majestry's Secret Service (week 5)_: for the given time T, we can determine (by max flow) whether all agent could to go shelters safely! So we use the __binary search__ to find the optimal time T.
* _Coin Tossing Tournament(week 6)_: In this problem, we try different decisons on the unknown turns. The final goal is to whether we could fill all the sinks. This could be turned to a flow problem and the maximum flow algorithm automatically finds the best decision.
  * maybe thought in this way: max flow corresponds to the particular strategy that is optimal; flow_2 corresponds to strategy_2; flow_3 corresponds to strategy3...
* _Shopping Trip (week 6)_: edge-disjoint s-t-paths <==> max flow from s to t
* ___Kingdom Defence(week 6)___: max flow with __lower bound__. 
* _Cantonal Courier_
* _Car Sharing()_ 

## Extra

* `odd - odd = even` , `even - even = even`
  - _even pairs(week 1)_ , _even matrices(week 1)_ , 
* others
  - _False Coin(week 1)_ : 
  - __Union-Find__ data structure. Methoded in lecture2 MST
  - Answer queries __on the fly__! See _Evoluation(week 2)_ 




[I'm an inline-style link](https://www.google.com)

---
**NOTE**

It works with almost all markdown flavours (the below blank line matters).

---

