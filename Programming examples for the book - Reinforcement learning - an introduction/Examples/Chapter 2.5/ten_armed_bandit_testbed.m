function [rewards,optimalActPercentage]=ten_armed_bandit_testbed(epsilon,bias)
%%10-armed bandit model
numSteps=1000;

%Generate the position of the ceneter of distribution for every arm 
arms=normrnd(zeros(10,2000),1);

numTimesUsed=zeros(10,2000);
Qt=repmat(bias,10,2000);

rewards=zeros(1000,1);
optimalActPercentage=zeros(1000,1,'double');

%Find best action for each bandit
optimalActions=zeros(2000,1,'double');
for step=1:2000
    [M,I] = max(arms(:,step));
    optimalActions(step)=I;
end;

for step=1:1000
    sumRewards=0;
    optActCount=0;
    for bandit=1:2000
        %selecting random or greedy action
        if rand<epsilon
           %select random action
           action=floor(rand*10)+1;
        else
            %select greedy action
            action=findBestAction(Qt(:,bandit));
        end
        reward= normrnd(arms(action,bandit),1);
        numTimesUsed(action,bandit)=numTimesUsed(action,bandit)+1;
        Qt(action,bandit)=Qt(action,bandit)+(1/numTimesUsed(action,bandit))*(reward-Qt(action,bandit));
        sumRewards=sumRewards+reward;
        if(action==optimalActions(bandit)) 
            optActCount=optActCount+1;
        end;
    end
    rewards(step)=sumRewards/2000;
    optimalActPercentage(step)=optActCount/2000;
    %Display the number of current step to see the progress
    if(mod(step,50)==0) 
        step 
    end
end


