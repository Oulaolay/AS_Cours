
  require 'nn'
  require 'gnuplot'
 require 'nngraph'
require 'dpnn'
 local DIMENSION=2
 local n_points=1000

 -- Tirage de deux gaussiennes
   local mean_positive=torch.Tensor(DIMENSION):fill(1);
   local var_positive=1.0
   local mean_negative=torch.Tensor(DIMENSION):fill(-1);
   local var_negative=1.0
   local xs=torch.Tensor(n_points,DIMENSION)
   local ys=torch.Tensor(n_points,1)
   for i=1,n_points/2 do
        xs[i]:copy(torch.randn(DIMENSION)*var_positive+mean_positive);
        ys[i][1]=1
   end

   for i=n_points/2+1,n_points do
       xs[i]:copy(torch.randn(DIMENSION)*var_negative+mean_negative);
       ys[i][1]=2
   end
   local x_train=xs
   local y_train=ys


   ---------------------------------------------
   -------------- PREPARE BATCH ----------------
   --------------NO VECTOR ACCURACY-------------
   --------------LOSS PRECISION ----------------
   ---------------------------------------------

   function Prepare_batch()
       batch_size=25
       --if(output[i]~=yhat[i]) then
         --local nb_batch=torch.round(x_train:size(1)/batch_size)+1
         shuffle = torch.randperm(x_train:size(1))
         x_batch = shuffle:chunk(batch_size,1)
         y_batch = shuffle:chunk(batch_size,1)
      --end
         return x_batch,y_batch,batch_size
   end


   function Accuracy_novect(y,out)
     for i=1,out:size(1) do
         if(out[i]*y[i]>0) then
           sum=sum+1
         end
     end
     acc=sum/out:size(1)
     return acc
   end

   function loss_precision(output,yhat)
  precision=0
  for i=1,output:size(1) do
  --  print(output[i])
    --print(yhat[i])
    break;
      if(output[i][1]==yhat[i][1]) then
        precision=precision+1
  end
end
  return precision/output:size(1)

end


--------------------------------------------
--------------TESTING ----------------------
--------------------------------------------

function Eval(xtest,ytest)
    output = model:forward(xtest)
    return Accuracy_vect(ytest,output)
end


--------------------------------------------
--------------CREATE MODEL ----------------------
--------------------------------------------
local model_1=nn.Sequential()
model_1:add(nn.Linear(DIMENSION,10))
model_1:add(nn.Sigmoid())
model_1:add(nn.Linear(10,10))
model_1:add(nn.ReinforceBernoulli())
model_1:add(nn.Linear(10,1))

---------------------------------------------
-------------- TRAINNING  -------------------
---------------------------------------------
local criterion=nn.MSECriterion()
model_1:reset(0,1)

local learning_rate=1e-3
local maxEpoch=300
local all_losses={}
for iteration=1,maxEpoch do
    x_batch,y_batch,batch_size=Prepare_batch()
    loss=0
    --acc=Eval(x_test,y_test)
    --table.insert(all_Eval,acc)
    for i=1,batch_size do
        model_1:zeroGradParameters()
            -- Transform Table to Long ---> index(dim,torch.LongTensor))

        output = model_1:forward(x_train:index(1,x_batch[i]:long()))
        loss=nn.PairwiseDistance(2):forward(output,y_train:index(1,y_batch[i]:long()))
        model_1:reinforce(loss-loss:mean())
        avg_loss=criterion:forward(output,y_train:index(1,y_batch[i]:long()))
        delta=criterion:backward(output,y_train:index(1,y_batch[i]:long()))
        model_1:backward(x_train:index(1,x_batch[i]:long()),delta)
        model_1:updateParameters(learning_rate)
    end
    table.insert(all_losses,avg_loss)

   gnuplot.plot('train loss',torch.Tensor(all_losses))

end
