
  require 'nn'
  require 'gnuplot'
 require 'nngraph'
 require 'distributions'

 local DIMENSION=1
 local n_points=1000
 local batch_dep=40


 -- Tirage d'une gaussiennes
   local mean_positive=4
   local var_positive=0.5
   local x_train_d=torch.Tensor(n_points,DIMENSION)

   for i=1,n_points do
        x_train_d[i]=distributions.norm.pdf(i*0.01,mean_positive,var_positive);
   end
   gnuplot.plot('hello',x_train_d)
   z_noise=torch.Tensor(batch_dep,DIMENSION)
   label_dat=torch.Tensor(batch_dep,DIMENSION)
   label_dat_false=torch.Tensor(batch_dep,DIMENSION)
   for i=1,batch_dep do
      z_noise[i]=torch.uniform(0,1)
   end
   for i=1,batch_dep/2 do
      label_dat[i]=1
   end
   for i=batch_dep/2+1,batch_dep do
     label_dat[i]=1
   end
   print(label_dat)
   for i=1,batch_dep do
     label_dat_false[i]=1
  end


--------------------------------------------
--------------CREATE MODEL G----------------------
--------------------------------------------
local model_g=nn.Sequential()
model_g:add(nn.Linear(DIMENSION,10))
model_g:add(nn.SoftPlus())
model_g:add(nn.Linear(10,1))


--------------------------------------------
--------------CREATE MODEL D----------------------
--------------------------------------------
local model_d=nn.Sequential()
model_d:add(nn.Linear(DIMENSION,10))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(10,10))
model_d:add(nn.ReLU())
model_d:add(nn.Linear(10,1))
model_d:add(nn.Sigmoid())



function Prepare_batch()
    batch_size=25
      shuffle = torch.randperm(x_train_d:size(1))
      x_batch_d = shuffle:chunk(batch_size,1)

      return x_batch_d,batch_size
end


---------------------------------------------
-------------- TRAINNING  -------------------
---------------------------------------------
local criterion=nn.BCECriterion()

local learning_rate=1e-3
local maxEpoch=1000
local all_losses={}
step=1
for iteration=1,maxEpoch do
    loss=0
    for k=1,step do
      x_batch_d,batch_size=Prepare_batch()
      loss_d=0
      loss_g=0
      for i=1,batch_size do
        model_d:zeroGradParameters()
        -- Forward noise
        data_gen=model_g:forward(z_noise)
        new_batch=x_train_d:index(1,x_batch_d[i]:long())
        new_batch:sub(batch_dep/2+1,batch_dep):copy(data_gen:sub(1,batch_dep/2))
        output=model_d:forward(new_batch)
        print(output)
          --print(output)
        --Explication pour la loss :
        --  1/n sum_i weights[i] * (t[i] * log(o[i]) + (1 - t[i]) * log(1 - o[i]))
        -- t=1 pour la descente de gradient
        loss_d=criterion:forward(output,label_dat)
        delta=criterion:backward(output,label_dat)
        model_d:backward(data_gen,delta)
      end
      table.insert(all_losses,loss_d)

    end

    x_batch,batch_size=Prepare_batch()

    for i=1,batch_size do
      -- Remise à zeros des parametres
      model_g:zeroGradParameters()
      model_d:zeroGradParameters()
      -- Forward noise
      data_gen=model_g:forward(z_noise)
      -- Forward data_gen in model D
      output=model_d:forward(data_gen)
      -- Forward avec le criterion BCE
      loss_g=criterion:forward(output,label_dat_false)
      delta=criterion:backward(output,label_dat_false)
      -- update for back_prop data_gen
      model_d:backward(data_gen,delta)
      -- update for back_prop generator
      model_g:backward(z_noise,delta)
    end






end
gnuplot.plot('train loss',torch.Tensor(all_losses))
