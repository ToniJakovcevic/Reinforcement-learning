##Q-learning: Off policy TD control
Python implementation of Cliff Walking example covered in section 6.5 of the book (second edition).

The results:
Action map for every state (the final policy) for Q-learning after 500 episodes. In this case the path from start to goal is optimal (walking right next to the cliff).


```
→→→→↓→↓→→→↓↓
→→→↓→→→↓→→→↓
→→→→→→→→→→→↓
↑↓↓↓↓↓↓↓↓↓↓↓
```

Action map for Sarsa after 500 episodes. It takes the safer path away from the cliff.

```
→→→→→→→→→→→↓
→↑↑↑↑→↑↑↑←↑↓
↑↑→↑←↑↑↑←←→↓
↑↓↓↓↓↓↓↓↓↓↓↓
```

