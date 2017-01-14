
require 'nn'
require 'gnuplot'
require 'nngraph'
require 'rlenvs'
require 'env'
require 'optim'
require 'dpnn'

local CartPole=require 'rlenvs.CartPole'
local env=CartPole()
local actionSpace = env:getActionSpace()
local etat= env:start()


function Prepare_batch(memory,action)
  --Attention utilisation du max pour les premieres occurences, sinon taille<batch_size
  discount_factor=0.5
  batch_size=math.min(table.getn(memory),25)

  y_batch=torch.Tensor(batch_size,actionSpace['n']):zero()
  x_batch=torch.Tensor(batch_size,table.getn(memory[1].etat)):zero()
  shuffle = torch.randperm(table.getn(memory))
  for i=1,batch_size do
    x_batch[i]=torch.Tensor(memory[shuffle[i]].etat)
    y_batch[i]=memory[shuffle[i]].reward+discount_factor*torch.max(torch.Tensor(memory[shuffle[i]].etat_suivant))
  end

  return x_batch,y_batch,batch_size
end

function Create_memory_table(action1,etat1,reward1,etat_suivant1,terminal1)
  new_memory={
    etat = etat1,
    action=action1,
    etat_suivant = etat_suivant1,
    reward = reward1,
    terminal = terminal1
  }
  return new_memory
end


local criterion=nn.MSECriterion()
--model_1:reset(0,1)

local learning_rate=0.1
local maxEpoch=150
local all_losses={}
local Length_memory=200 --Taille de la mémoire
local Memory={}
T=50
M=100
local eps=0.1

local model_1 = nn.Sequential()
model_1:add(nn.Linear(4,10))
model_1:add(nn.Tanh())
model_1:add(nn.Linear(10, 10))
model_1:add(nn.Tanh())
model_1:add(nn.Linear(10, 10))
model_1:add(nn.Tanh())
model_1:add(nn.Linear(10, actionSpace['n']))

for episode=1,M do
  for t=1,T do
    loss=0
    if(torch.uniform()>eps) then
      action=math.random(0,actionSpace['n']-1)
    end
    reward, etat_suivant, terminal = env:step(action)
    new_memory=Create_memory_table(action,etat,reward,etat_suivant,terminal)
    if(table.getn(Memory)~=200) then
      table.insert(Memory,new_memory)
    else
      table.remove(Memory,1)
    end
    x_batch,y_batch,batch_size=Prepare_batch(Memory)
    for i=1,batch_size do
      model_1:zeroGradParameters()
      output=model_1:forward(x_batch[i])
      loss=loss+criterion:forward(output,y_batch[i])
      delta=criterion:backward(output,y_batch[i])
      model_1:backward(x_batch[i],delta)
      model_1:updateParameters(learning_rate)
    end
    table.insert(all_losses,loss)

  end
  gnuplot.plot('train loss',torch.Tensor(all_losses))

end
