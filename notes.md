# ETH ALGORITHM LAB 2016 FALL



## Dynamic Programming

__Reference: [Slide of Stanford CS 97SI](https://web.stanford.edu/class/cs97si/)__

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
Graph DP Example
Problem: Odd Route in Week 10
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

* _A New Hope(week 6)_: Tree DP. 

* _Odd Route(week 10)_: could be considered as Graph DP

* _Bonus Level(week 10):_ 2-Dimension DP, but with two players. 

* _Connecting Cities(week 11)_: Tree DP, __find the maximum matching on a tree__ in O(n).

* _Punch(week 11)_: 1-D DP, with some additional informations (which kind of beverage is used). The subproblem is MinCosts[i] which is the min cost for buying no less than i liters beverage. And we finally output MinCost[k].

* ___New Tiles(week 12)___: 1- dimension DP, but each subproblem is also a 1-D DP.

* ___Corbusier's Modulor(week 12)___: 2-D DP. "whether exists some elements that sum to k". ```M[i][j]```=1 if first(i) elements that could sum to j; and ```M[i][j]``` =0 vice versa.



## Binary Search

> Goal: find the smallest k that is ‘large enough’.
>
> Properties: (1) for a fixed k, you can check __efficiently__ if it is ‘large enough’, (2) The property of not being too small is __monotone__. [Slide 2]

```c++
Binary Search Outline
int lmin = 0, lmax = 1;
while (too_small(lmax)) lmax *= 2;
while (lmin != lmax){
  int p = (lmin + lmax) / 2;
  if (too_small(p))
    lmin = p+1;
  else
    lmax = p;
}
L = lmin;
```

```c++
Binary Search too_big() version
int lmin = 1;
int lmax = n;
while(lmin != lmax){
  int p = (lmin + lmax + 1) / 2; // different with the too_small() version
  if(too_big(p))
    lmax = p - 1;
  else
    lmin = p;
}
```



#### problems:

* _Evoluation(week2)_
* _TheeV(week 4)_: sorted cities with ascending distance to the first TV transmitter. Binary search the position with splits cities into two parts.
* _Revenge of the Sith (week 10)_: Binary search the largest k, and for each candidate k, we need to construct a new sub DT. 



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

> Split all elements to different subsets (e.g. S1=used and S2= not used; e.g. S1=elements in the first room, S2=elements in the 2nd room, S3=elements in the 3rd room) and examine properties that each subset should satisfy.



> __Example problem: Is there a subset of S which sums to k? (if k is small, it could be solved by DP withO(n*k) ).__ 
>
> Search all possible strategies: in each strategy, we make a decision for each item that whether or not to take this item. ==> n is small O(n*2^(n/2)), n<=40.
>
> _Another approach:_ Dynamic Programming. 



> Methods: __Backtracking__ V.S. __Representing set as bits__ [Slide 5]
>
> Trick: complexity should allow exponential time.
>
> Trick2: split and list



#### Problems:

* _Light at the Museum(week 5)_: Similar to the example problem, but need to check __M constraints__. 
* _Planks (week 11)_: 
  * Split all elements to four subgroups
  * Examine the sum (the property each subgroup should satisfy) of each subgroup.

## CGAL - Intro

> Combinatorial Algorithm <— Geometric Predicate <— Algebraic Computation
>
> 1.  algebraic computation is non-trivial 
>     2. Filtering to ganrantee correctness: Check whether things go fine and use exact algebra only 	when needed
> 2.  Exact construction is very expensive ==> If possible, use predicates with numerical tests decide branching instead of exact construction

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

> Linear optimization: Optimal solution, Unbounded, Infeasible
>
> LP ex1: cancer therapy -> lifting space
>
> LP ex2: seperated by polynomial of d?
>
> QP: matrix D sould be positive semidefinite
>
> QP: solving nonnegative quadratic program is faster

```c++
choice of exact internal number type
#include <CGAL/Gmpz.h> : for integer 
#include <CGAL/Gmpzf.h>: for float number
```

```c++
// Without Bland Pivot Rule
typedef CGAL::Quadratic_program<int> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

Program lp (CGAL::SMALLER, true, 0, false, 0);
Solution s = CGAL::solve_linear_program(lp, ET());
assert (s.solves_linear_program(lp));

// With Bland Pivot Rule
typedef CGAL::Quadratic_program<ET> Program;
typedef CGAL::Quadratic_program_solution<ET> Solution;

Program lp(CGAL::SMALLER, false, 0, false, 0);
CGAL::Quadratic_program_options options;
options.set_pricing_strategy(CGAL::QP_BLAND);
Solution s = CGAL::solve_linear_program(lp, ET(), options);
assert (s.solves_linear_program(lp));
```



#### Problems:

* _Diet(week 7)_: linear programming
* _Portfolios(week 7)_: similar to the example on slide, but add an addition constraint.
* _Inball(week 7)_: LP. [Solution here](https://github.com/chocolates/ETH-Algorithm-Lab/blob/master/Official%20Solutions/solution-inball.pdf).
* _Stamp Exhibition(week 8)_: use Gmpzf. There are 200 variables.
* ___Radiation Therapy(week 12)___: d-degree polynomial with THREE variables: choose 3 elements in (d+3) ==> if d==30, then there are about 5000~ variables.
* ___The Empire Strikes Back(week 12)___: 



## Delaunay Triangulation

> An __empty disk of maximal radius__ passes through three points, if its center is inside the convex hull. These __maximal empty disks__ collectively define what is called the __Delaunay Triangulation__.
>
> Properties of DT:
>
> > 1. It maximizes the smallest angle.
> > 2. It contains __Euclidean Minimum Spanning Tree__.  ==> related with the connection problems.
> > 3. It contains __nearest neighbor graph__. (The edge between each node to its nearest neighbor is in DT)
> > 4. For the __second nearest neighbor__: let v1 be the nearest neighbor of v and v2 be the second nearest neighbor of v. Then at least one of edge (v,v2) or edge(v1, v2) is in DT.
> > 5. It is unique in general.
> > 6. It could be construct efficiently! __O(n*logn)__ in 2D. ==> when we want to change time complexity from n^2 to nlogn
> > 7. Delaunay Triangulation v.s. Voronoi-Diagram.  ==> ```t.nearest_vertex()```



```c++
DT Examplary Calls 1
  
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K> Triangulation;
typedef Triangulation::Finite_faces_iterator Face_iterator;
typedef Triangulation::Finite_vertices_iterator Vertex_iterator;

Triangulation t;
t.insert(pts.begin(), pts.end());
t.nearest_vertex(K::Point_2 p);

// iterate all faces
for(Face_iterator f=t.finite_faces_begin(); f != t.finite_faces_end(); f++){
  K::Point_2 p = t.dual(f); // the center of the circle of face f
  Triangulation::Vertex_handle v1 = f -> vertex(1); // the vertex
  Triangulatoin::Face_handle f_neighbor = f -> neighbor(1); // the neighbor face
}

// iterate all edges connected to node v
Triangulation::Vertex_handle v;
Triangulation::Edge_circulator c = t.incident_edges(v);
do {
  if (t.is_infinite(c)){
    ...
  }
}while (++c != t.incident_edges(v));

// find the nearest neighbor distance for each vertex
for(Vertex_iterator v = t.finite_vertices_begin(); v != t.finite_vertices_end(); v++){
  Triangulation::Edge_circulator edge = t.incident_edges(v);
  do{
	if( ! t.is_infinite(edge)){
	    K::Segment_2 seg = t.segment(edge);
	    K::FT dist = seg.squared_length();
	    K::FT squared_radio = dist / 4;
	    if(count_edges==0){
	        nearest_distance_square = squared_radio;
	        count_edges = 1;
	    }
	    else{
	        if( squared_radio < nearest_distance_square )
	            nearest_distance_square = squared_radio;
	    }
	}
 } while(++edge != t.incident_edges(v));
}
```



reference: [CGAL doc 2D triangulation with vertex information](https://judge.inf.ethz.ch/doc/cgal/doc_html/Triangulation_2/Triangulation_2_2info_insert_with_pair_iterator_2_8cpp-example.html#_a1)

```c++
DT Example Calls 2 (with info())
  
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel         K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K>    Vb;
typedef CGAL::Triangulation_face_base_with_info_2<int, K>      Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>                    Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds>                      Delaunay;
typedef Delaunay::Finite_edges_iterator							Edge_iterator;
typedef Delaunay::Finite_faces_iterator 						Face_iterator;
typedef Delaunay::Vertex_handle									Vh;

Delaunay t;
vector<std::pair<K::Point_2, int> > planets_pos(n);
for(int i=0; i<n; i++){
		K::Point_2 p;
		cin >> p;
		planets_pos[i] = make_pair(p, i);
	}
t.insert(planets_pos.begin(), planets_pos.end());
for(Edge_iterator e=t.finite_edges_begin(); e != t.finite_edges_end(); e++){
  Vh v1 = e -> first -> vertex((e -> second + 1) % 3);
  Vh v2 = e -> first -> vertex((e -> second + 2) % 3);
  int id1 = v1 -> info();
  int id2 = v2 -> info();
  double squared_dist = CGAL::to_double(CGAL::squared_distance(v1 -> point(), v2 -> point()));
}
```





#### Problems:

* _Graypes(week 8)_: DT contains __nearest neighbor graph__.
* _Bistro(week 8)_: __nearest neighbour__
* ___H1N1(week 8)___: moving the disk without colliding with a given point set P. __DT + BFS__ but check the distance between user and its nearest neighbor at beginning.
* _Germs(week 8)_: nearest neighbor graph + sort/binary search
* _Light the Stage(week 10)_: Use DT for the first test case. For the second test case, just try the trivial method.
* _Revenge of the Sith(week 10)_: It uses the property that __DT contains the EMST__. __Given the distance threshold, two nodes are connected on the EMST of G <==> these two nodes are connected on G__. 
* _[Clues(week 11)](https://github.com/chocolates/ETH-Algorithm-Lab/blob/master/Official%20Solutions/solution-clues.pdf)_: Connected components —> for the given distance threshold, EMST and original graph is equal w.r.t connectivity. Furthermore, to check whether the graph is two colorable, we just need to construct two new DT.
* _Snakes strike back(week 12)_: very similar to _H1N1_. Four vertices of each cage are inserted the Triangulation.

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
  * For each node, there are (even_num_edges, odd_num_edges) × (even_weight, odd_weight) __four different states__, so __split each original vertex to four vertices__. Then use __Dijkstra shortest path__ find shortest path between the new source and target. It may also considered in DP approach.
* _Domino Snake(week 12)_: 
  * Chessboard —> two states
  * connected or not
  * "whether such path exist or not" <==> "whether the entrance and exit are in different states of chessboard && whether entrance and exit are connected"





> 2. Basic Graph Algorithms
>    - Distance: Dijkstra shortest paths, Prim minimum spanning tree, Kruskal minimum spanning tree.
>    - Components: Connected, biconnected, strongly connected components
>    - Matchings: General unweighted matching
>    - Others: topological sorting; Eulerian tours; Union-Find (e.g. _GoldenEye_).
> 3. Dijkstra shortest path __complexity__: ```O(V*logV + E)```, Dijkstra's algorithm finds all the shortest paths from the __source vertex__ to __every other vertex__.
> 4. Strongly connected components complexity: ```O(v+E)```. 
> 5. Biconnected subgraph: the biconnected components of a graph are the maximal subsets of vertices such that the __removal of a vertex__ from a particular component __will not__ disconnect the component. Complexity ```O(V+E)```
> 6. Prim MST: add the closest undiscovered neighbor of all discovered neighbors; Kruskal MST: add the next shortest edge without creating a cycle 
> 7. BGL __does not__ provide __weighted matching algorithms__.

```c++
//Union-Find in C++ (BGL connected... -> Incremental Connected Components -> Disjoint set)
#include <boost/pending/disjoint_sets.hpp>
typedef boost::disjoint_sets_with_storage<> Uf;
Uf ufa(num_elements);
ufa.find_set(i) // return the representation of i
ufa.union_set(i, j) // merge 
```

```c++
// Typological Sort one: call bgl topological_sort() function.
```

```c++
//Topological Sorting two: Kahn'a algorithm
Initialization: 
	L <- Empty queue // queue that will contain all sorted elements
    S <- Queue containing all elements that has no incoming edges // queue that contains all elements that have no incoming edges
    while(S is not empty){
      n <- S.pop();
      L.add(n);
      for m that there is a link from n to m:
      	remove edge(n, m) from graph;
      	if(m has no incoming edges)
          S.add(m)
    }
	if Graph still have edges:
		Graph has cycle
    else
      	return L
```

```c++
//Topological Sorting three: DFS (same in BGL)
void topologicalSort(){
  stack<int> Stack;
  vector<int> visited(V, false);
  for (int i = 0; i < V; i++)
    if (visited[i] == false)
      topologicalSortUtil(i, visited, Stack);
}
void topologicalSortUtil(int v, vector<int>& visited, stack<int> Stack){
  visited[v] = true;
  for (i = adj[v].begin(); i != adj[v].end(); ++i)
    if (!visited[*i])
      topologicalSortUtil(*i, visited, Stack);
  Stack.push(v);
}
```



#### problems:

* ___GoldenEye(week 12)___: For the given distance threshold, two nodes are connected on the MST is equal to that these two nodes are connected on the original graph. Furthermore, use the __Union-Find__ data structure.
* _Ant Challenge(week 4)_: 
* _Important Bridges(week 4)_: biconnected connected graph
* _Buddy Selection(week 4)_: maximum matching on unweighted graph.
* _Connecting Cities (First subproblem in week 11)_: find __maximum matching on a tree__.  Similar to the subproblem _Downhill Course in Winter games_, that find __maximum independent set on a graph whose max degree is 2__.
* ___Bob's Burden(week 13)___: find the central ball.


```c++
/* Some useful stuff in BGL */
// whether an edge exist or not
tie(e, success) = boost::edge(v1, v2, G0);
if(!success){
  tie(e, success) = add_edge(v1, v2, G0);
}
// number of vertices
num_vertices(G); // in the tutorial
// Iteration on edges and vertices --> see tutorial
// 
```






## Max Flow / Max Flow Min Cost

> The MaxFlow (MinCost) problem
>
> - some quantities are reserved in some sources. These quantities want to go to the sinks.
> - node v: the abstract states. For example, each node in _Consecutive Constructions_ (city, in/out)
> - edge (u, v): the quantity can go from u (abstract state) to v (abstract state)
> - different decisions <==> different path for the quantities. In more details, in most cases, we have decision variables and the objective function. What we want to do is to choose the particular decision that make objective function optimal. When we change decisions, we change the network structure for the flow (e.g. _On Her Majesty's Secret Service_) or change the capacity of edges. This results in different flow!
> - final distribution of quantities &/ distribution of flow is related with the objective function
> - max flow algorithm  => choose the routes that maximize flow (min cost) ==> this should correspond to the optimal decision.

```
Push Relabel is almost always the best choice in BGL. O(n^3)
```

> Common tricks:
>
> 1. __Vertex capacity__: break this vertex v to two vertex v1 and v2, all in edges connected to v1 and all out edges linked on v2 and there is a edge from v1 to v2 with unlimit capacity.
> 2. __Minimum flow per edge__: see the [solution of Kingdom Defence](https://github.com/chocolates/ETH-Algorithm-Lab/blob/master/Official%20Solutions/solution-kingdomdefence.pdf).
> 3. With edge capacity set to 1, the maximum number of __edge-disjoint s-t-paths__ is equal to the maximum flow from s to t. And these three quantities are equal:
>    1. the minimum number of edges separating u and v
>    2. the maximum number of edge-disjoint u-v paths // 1=2 is the Corollary of Menger's Theorem
>    3. the maximum flow from u to v
>    4. the minimum cut between u and v (if edge capacity is set to 1's)
> 4. For the MaxFlowMinCost problem, we tend to make the costs negative to speed up the programm (pass test data!). Because successive_shortest_path_nonnegative_weights() is faster than cycle_canceling(). A common trick is to [shift the cost of each edge in such a way that each s-t-path gets shifted by the same total amount](https://github.com/chocolates/ETH-Algorithm-Lab/blob/master/Official%20Solutions/carsharing-solution.pdf). 

#### Problems:

* _Real Estate Market(week 9)_: Max Flow Min Cost. Make the costs nonnegative.
* _Satellites(week 9)_: Minimum vertex cover on bipartite graph. -> max flow then BFS.
* _Algocoon Group(week 9)_: Global [Min-Cut](http://theory.stanford.edu/~trevisan/cs261/lecture13.pdf). Following steps:
  * choose an arbitrary vertex s in V
  * for each t \in V - {v}, compute the minimum s-t cut
  * return the minimum one among all s-t minimum cut.
* _Canteen(week 9)_: Max Flow Min Cost, still make all costs on edges nonnegative. __Shift the cost of each edge in such a way that each s-t-path gets shifted by the same total amount__ 
* _Consecutive Construction (week 11)_: 
* _Missing Roads(week 11)_: Different flow corresponds to different choice (strategy). The Max Flow should correspond to the optimal strategy. In other words, the set of all possible flows and the set of all strategies should have one-to-one map. 


---

__Problems related with flow__

* ___Max Flow Min Cut Theorem___: maximum amount of flow passing from the source to the sink is equal to the total weight of the edges in the minimum cut, i.e. the smallest total weight of the edges which if removed would disconnect the source from the sink.
  * Finding the cut: BFS on the residual graph starting at source.
* ___Menger's Theorem___ : The maximum number of __vertex-disjoint S-T paths__ is equal to the minimum size of __S-T separating vertex set__.
* ___Minimum Vertex Cover___ and ___Maximum Independent Set___ are complementary. They are NP - complement in general graph. But, in __bipartite graph__, according to ___König Theorem___, the number of edges in a maximum matching is equal to the number of vertices in a minimum vertex cover (= n - size of maximum independent set). In __other special cases__, may be trivial (greedy method is ok)

---

#### Problems:

* _On her Majestry's Secret Service (week 5)_: for the given time T, we can determine (by max flow) whether all agent could to go shelters safely! So we use the __binary search__ to find the optimal time T.
* _Coin Tossing Tournament(week 6)_: In this problem, we try different decisons on the unknown turns. The final goal is to whether we could fill all the sinks. This could be turned to a flow problem and the maximum flow algorithm automatically finds the best decision.
  * maybe thought in this way: max flow corresponds to the particular strategy that is optimal; flow_2 corresponds to strategy_2; flow_3 corresponds to strategy3...
* _Shopping Trip (week 6)_: __edge-disjoint s-t-paths__ <==> max flow from s to t
* ___Kingdom Defence(week 6)___: max flow with __lower bound__. 
* _Knights(week 7)_: max flow with __vertex capacity__: split each vertex to two vertices
* _Casino Royale (week 9)_: Similar to Car Sharing problem. __Lots of binary variable (choices), which corresponds to different paths for the flow!__ 
* _First problem in Winter Games_. Find (greedy method) the __maximum independent set__ in a graph with __max degree no more than 2__.
* _Consecutive Constructions(week 11)_: __maximum matching on a biparitite graph__
* ___[Car Sharing](https://github.com/chocolates/ETH-Algorithm-Lab/blob/master/Official%20Solutions/carsharing-solution.pdf)___(week 11): 
  * We construct the graph following given constraints. __Important__: what are the nodes(abstract states)
  * Different strategies correspond to different paths of the flow.
  * Use the so-called coordinate compression trick.
  * Eleminate negative costs by making each s-t path get shifted by the same total amount!


* _Placing Knights (week 12)_: Bipartite graph! Maximum independent set!
* Cantonal Courier_

## Extra

* `odd - odd = even` , `even - even = even`
  - _even pairs(week 1)_ , _even matrices(week 1)_ , 

* others
  - _False Coin(week 1)_ : 

  - __Union-Find__ data structure. Mentioned in lecture2 MST

    ```c++
    #include <boost/pending/disjoint_sets.hpp>
    typedef boost::disjoint_sets_with_storage<> Uf;
    Uf ufa(num_elements);
    ufa.find_set(i) //
    ufa.union_set(i, j)
    ```

    ​


- Answer queries __on the fly__! See _Evoluation(week 2)_ 

* c++ MISC.

  ``` c++
  typedef struct{
    int len;
    int ring_position;
  } boat;
  boat boat_variable; // 
  ```

* Both the material from tutorial and the collection of problems form this course. Solving a problem in following way:

  * understand the task (key concept, data structure, techniques and skills are covered in the tutorial and/or practiced in a problem. E.g. Union-Find) Could check the examples to see if they concur with your understanding.
  * find an appropriate model (__modelling__)
  * design efficient algorithm (__algorithm design__)
  * Avoid "stupid" mistakes (e.g. read all inputs; std::ios_base::sync_with_stdio(false); )

* DT others

  * Problem: given a finite discrete data set and a distance delta, find all pairs of points (p, q) with distance |p-q| < = delta. 
    * construct DT
    * iterate each edge on the DT and mark those whose distance is <= delta. Construct a graph using these marked edges
    * Start on each point on the constructed graph, do a DFS and stop each branch of the search at the first point q such that |p-q|>delta.


---

__Problems to revisit__

* _H1N1_ : add vertex information and face information 
* Odd route: use shortest path, not DP.
* _Connecting Cities_: Find maximum matching on a tree.
* _New tiles._ 1-d DP, but subproblem is also a 1-D DP.

---

