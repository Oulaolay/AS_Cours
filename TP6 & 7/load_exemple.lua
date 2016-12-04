require 'torch'
require 'model_utils'
require 'nn'
require 'nngraph'
require 'gnuplot'


---------------------------------------------
-------------- PARAMETERS  ------------------
---------------------------------------------

local nb_char_seq=5 -- Nombre de caractère par séquence
n=100
N=100 --Dimension latente
local batch_size=100
local utils=require 'utils'
local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'
local model_utils=require 'model_utils'
local v=CharLMMinibatchLoader.create("data.t7","vocab.t7",batch_size,nb_char_seq)
local V_size=v.vocab_size


length_table=table.getn(v.x_batches)
local split=torch.round(length_table*75/100)+1
shuffle=torch.randperm(split)



---------------------------------------------
-------------- DATA, MINI BATCH -------------
---------------------------------------------


function create_data()
  local X_train={}
  local Y_train={}
  local X_test={}
  local Y_test={}

  for i=1,split do
    X_train[i]=v.x_batches[i]
    Y_train[i]=v.y_batches[i]
    X_test[i]=v.x_batches[split+i]
    Y_test[i]=v.y_batches[split+1]
  end

  return X_train,Y_train,X_test,Y_test
end

function Prepare_batch(X_train_i, Y_train_i)

   local x_batch = {}
   local y_batch = {}

   for j=1, nb_char_seq do
      table.insert(x_batch,torch.zeros(batch_size,V_size))
      table.insert(y_batch,torch.Tensor(batch_size))
   end

   for i = 1,batch_size do
      for j=1,nb_char_seq do
         x_batch[j][i][X_train_i[i][j]] = 1
         y_batch[j][i] = Y_train_i[i][j]
      end
   end
   return x_batch, y_batch
end
---------------------------------------------
-------------- CREATE MODEL -----------------
---------------------------------------------

criterion = nn.ParallelCriterion()
for i=1,nb_char_seq do
	criterion:add(nn.ClassNLLCriterion())
end
--Modele pour h, Utilisation de GRU :
--Création de noeud d'entrée, utilisation en tant que "parent"
ht_1=nn.Identity()()
x=nn.Identity()()
-- Implémentation des modules
Nr=nn.Linear(V_size,N)(x)
Nz=nn.Linear(V_size,N)(x)
Ur=nn.Linear(N,N)(ht_1)
Uz=nn.Linear(N,N)(ht_1)
sr=nn.CAddTable()({Ur,Nr})
sz=nn.CAddTable()({Uz,Nz})
rt=nn.Sigmoid()(sr)
zt=nn.Sigmoid()(sz)
rt_2=nn.CMulTable()({rt,ht_1})
rt_3=nn.Linear(N,N)(rt_2)
ht_tilde=nn.Tanh()(nn.CAddTable()({rt_3,nn.Linear(V_size,N)(x)}))
l_zt=nn.AddConstant(1)(nn.MulConstant(-1)(zt))
ht = nn.CAddTable()({nn.CMulTable()({ht_tilde,zt}),nn.CMulTable()({ht_1,l_zt})})

model_h=nn.gModule({x,ht_1},{ht})


--modele pour g :
i1=nn.Identity()()
lin=nn.Linear(N,V_size)(i1)
soft = nn.LogSoftMax()(lin)

model_g=nn.gModule({i1},{soft})

Gtheta=model_utils.clone_many_times(model_g,nb_char_seq)
Htheta=model_utils.clone_many_times(model_h,nb_char_seq)

--------------------------------------------
---------TESTING, CONCAT TABLE -------------
--------------------------------------------

--function Accuracy_vect(y,out)
  --  cumul=torch.cmul(torch.sign(y),torch.sign(out))
  --  acc=torch.mean(cumul)
  --  return acc
--end


--function Eval(xtest,ytest)
--    output = model:forward(xtest)
--    return Accuracy_vect(ytest,output)
--end


--Concatenation of tables in Lua
function TableConcat(t1,t2)
    for i=1,#t2 do
        t1[#t1+1] = t2[i]
    end
    return t1
end

---------------------------------------------
-------------- TRAINNING  -------------------
---------------------------------------------

all_losses={}
inputs={}
outputs={}
z=nn.Identity()()
inputs[1]=z
for i=1,nb_char_seq do
	inputs[i+1]=nn.Identity()()
	z=Htheta[i]({inputs[i+1],z})
	outputs[i]=Gtheta[i](z)
end
model=nn.gModule(inputs,outputs)



local nbIter=10
local lr=0.001
local all_losses={}
local all_Eval={}
X_train,Y_train,X_test,Y_test=create_data()



for iteration=1,nbIter do
    --acc=Eval(X_test,Y_test)
    --table.insert(all_Eval,acc)
    for j = 1,split do
      x_batch, y_batch = Prepare_batch(X_train[shuffle[j]], Y_train[shuffle[j]])

      local matrix_ones = torch.zeros(100,100)
      local input=TableConcat({matrix_ones},x_batch)

        --Initialisation Gradient
        model:zeroGradParameters()
        -- Forward partie
        output=model:forward(input)
        loss=criterion:forward(output,y_batch)
        table.insert(all_losses,loss)
        -- Backward partie
        delta=criterion:backward(output,y_batch)
        model:backward(input,delta)
        model:updateParameters(lr)
      end

      --if (iteration%100==0) then
      --    print('accuracy : ' , acc)
      --end

      gnuplot.plot(torch.Tensor(all_losses))
      --graph.dot(model.fg, 'MLP', 'MLP') ]] --

end
