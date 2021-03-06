##Blackjack - Monte Carlo Control with Exploring Starts
Monte Carlo control with exploring starts - Python implementation of Blackjack example covered in section 5.3 of the book (second edition).


Results after 5,000,000 episodes:

(Note: Forgot to put axis labels. Axis with range 1-10 represents the dealers first card, and the axis with range 1-21 represents the sum of cards in players hand.)

Policy (with usable ace)
![image](policy_usable_ace.png)

Policy (without usable ace)
![image](policy_no_usable_ace.png)

State-value function computed from action-value function (with usable ace)
![image](state_value_usable_ace.png)

State-value function computed from action-value function (without usable ace)
![image](state_value_no_usable_ace.png)

Action-value function for all states for action stick
![image](action_value_for_action_stick_with_no_usable_ace.png)

Action-value function for all states for action hit
![image](action_value_for_action_hit_with_no_usable_ace.png)
