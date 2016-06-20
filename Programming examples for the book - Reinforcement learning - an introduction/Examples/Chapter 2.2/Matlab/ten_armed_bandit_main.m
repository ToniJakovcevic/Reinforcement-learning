

%greedy
rewards=ten_armed_bandit_testbed(0);
plot(rewards);
hold on;
%epsilon=0.01;
rewards=ten_armed_bandit_testbed(0.01);
plot(rewards,'r');
hold on;
%epsilon=0.1;
rewards=ten_armed_bandit_testbed(0.1);
plot(rewards,'g');
hold off;
legend('greedy','epsilon=0.01','epsilon=0.1');
