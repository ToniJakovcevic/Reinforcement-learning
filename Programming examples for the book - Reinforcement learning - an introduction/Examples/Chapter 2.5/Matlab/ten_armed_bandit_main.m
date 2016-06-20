

%optimistic greedy
[rewards1,optimalActPercentage1]=ten_armed_bandit_testbed(0,5);

%realistic epsilon greedy, epsilon=0.01;
[rewards2,optimalActPercentage2]=ten_armed_bandit_testbed(0.01,0);

plot(rewards1);
hold on;
plot(rewards2,'r');
hold off;
legend('optimistic greedy','realistic epsilon=0.1');
xlabel('Steps') % x-axis label
ylabel('Average reward') % y-axis label

figure 

plot(optimalActPercentage1);
hold on;
plot(optimalActPercentage2,'r');
hold off;
legend('optimistic greedy','realistic epsilon=0.1');
xlabel('Steps') % x-axis label
ylabel('% Optimal action') % y-axis label
