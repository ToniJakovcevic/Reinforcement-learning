function action=findBestAction(Qt)
%Returns the index of the best action
%If there are several actions with the same Qt it breaks the ties randomly

max=Qt(1);
indices=[1];

for i=2:10
    if (Qt(i)>max)
        max=Qt(i);
        indices=i;
    elseif (Qt(i)==max)
        indices=[indices i];   
    end
end

if(size(indices,2)==1)
    action=indices(1);
else
    action=indices(floor(rand*size(indices,2)+1));
end