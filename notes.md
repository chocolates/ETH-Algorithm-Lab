# ETH ALGORITHM LAB 2016 FALL



## Dynamic Programming

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

## Brute Force / Split and List



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

