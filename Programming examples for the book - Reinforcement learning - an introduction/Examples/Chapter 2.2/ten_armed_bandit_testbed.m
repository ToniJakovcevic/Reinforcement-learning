function rewards=ten_armed_bandit_testbed(epsilon)
%%10-armed bandit model
numSteps=1000;

%Generate the position of the ceneter of distribution for every arm 
arms=rand(10,2000);
arms=arms*4-2; %Positioned randomly int the interval [-2,2]

numTimesUsed=zeros(10,2000);
Qt=zeros(10,2000);

rewards=zeros(1000,1);
for step=1:1000
    sumRewards=0;
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
    end
    rewards(step)=sumRewards/2000;
    %Display the number of current step to see the progress
    if(mod(step,50)==0) 
        step 
    end
end


