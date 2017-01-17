# ETH ALGORITHM LAB 2016 FALL



## Dynamic Programming

## Binary Search

## Sliding Window

> Iterate all elements and update the `candidate_state_variable` on the fly. Return the optimal `state_variable`.

> Two key components: (1) a state variable `state_variable` (e.g. the given number in _Deck of Cards_ and the fartest position in _Dominoes_) and (2) a current(interesting) interval `[l, r]`.  We slide the inteval from left to right and update the `state_variable`.  __IMPORTANT: what is the relationship between `state_variable` and sliding window__   Finally, return the optimal `state_variable`

> Special case, only one element in the sliding window (Iterate all elements). E.g. _Deck of Cards_ 

> The problem should be solved in linear time. Maybe need sort at beginning. 

#### Problems:

* _Dominoes (week1)_  Variable `pos`: the farthest position of domino that would fall down. Iterator each domino check (1) whether this dominon would fall down and (2) if this domino falls down, update `pos`
* _Deck of Cards(week 1)_ For the given `k`, maintain a state variable `current_sum` for candidate and a sliding window. Update `current_sum` when current window changes.
* _Attack of the clones(week 3)_  Variable `count_Jedi_number`: the number of Jedi that protect this position. Each element is either a `start_of_Jedi` or `end_of_Jedi`. Iterate each element, `count_Jedi_number++` if meet a `start_of_Jedi` or  `count_Jedi_number--` if meet a `end_of_Jedi`. 
* _Search Snippets(week 2)_ 

## Btute Force / Split and List

## LP/QP

## Delaunay Triangulation

## BGL

## Extra

* `odd - odd = even` , `even - even = even`
  - _even pairs(week 1)_ , _even matrices(week 1)_ , 
* others
  - _False Coin(week 1)_ : 





