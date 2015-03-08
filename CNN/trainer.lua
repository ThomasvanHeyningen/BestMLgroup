require 'torch'
require 'nn'
require 'globals'
require 'load_data'
require 'nn'
require 'optim'


function train(model)
    if unsup then
        return trainUnsup(model)
    else
        return trainSup(model)
    end
end


function trainUnsup(module)
    -- get all parameters
    x,dl_dx,ddl_ddx = module:getParameters()
    -- training errors
    local err = 0
    local iter = 0
    
    for t = 1,globals.epochSize,globals.batchSize do
        iter = iter+1
        xlua.progress(iter, globals.statinterval) 
        --------------------------------------------------------------------
        -- create mini-batch
        -- 
        local inputs = load_data.getBatch(globals.batchSize, true, false)
        local targets = inputs
        local n = (#inputs):totable()[1]
        --------------------------------------------------------------------
        -- define eval closure
        -- 
        local feval = function()
        collectgarbage()
            -- reset gradient/f
            local f = 0
            dl_dx:zero()
            -- estimate f and gradients, for minibatch
            for i = 1,n do
                -- f
                f = f + module:updateOutput(inputs[i], targets[i])
                -- gradients
                module:updateGradInput(inputs[i], targets[i])
                module:accGradParameters(inputs[i], targets[i])
            end
            -- normalize
            dl_dx:div(n)
            f = f/n
            -- return f and df/dx
            return f,dl_dx
        end
        --------------------------------------------------------------------
        -- one SGD step
        -- 
        sgdconf = sgdconf or {learningRate = globals.learningrate,
        learningRateDecay = globals.decay,
        learningRates = etas,
        momentum = globals.momentum}
        _,fs = optim.sgd(feval, x, sgdconf)
        err = err + fs[1]
        -- normalize
        -- module:normalize()
        --------------------------------------------------------------------
        -- compute statistics / report error
        -- 
        if math.fmod(t, globals.statinterval) == 0 then
            -- report
            print('==> iteration = ' .. t .. ', average loss = ' .. err/globals.statinterval)
             -- get weights
            eweight = module.encoder.modules[1].weight
            if module.decoder.D then
                dweight = module.decoder.D.weight
            else
                dweight = module.decoder.modules[1].weight
            end
            -- render filters
            --[[ dd = image.toDisplayTensor{input=dweight,
            padding=2,
            nrow=math.floor(math.sqrt(params.nfiltersout)),
            symmetric=true}
            de = image.toDisplayTensor{input=eweight,
            padding=2,
            nrow=math.floor(math.sqrt(params.nfiltersout)),
            symmetric=true}
            --]]

            -- live display
            -- save stuff
            -- image.save(globals.mainDir .. '/filters_dec_' .. t .. '.jpg', dd)
            -- image.save(globals.mainDir .. '/filters_enc_' .. t .. '.jpg', de)
            -- torch.save(globals.mainDir .. '/model_' .. t .. '.bin', module)
            -- reset counters
            err = 0; iter = 0
        end
        
    
    end
    return module
end




function trainSup(model)
    trainLogger = optim.Logger(paths.concat(globals.saveDir, 'train.log'))
    if model then
        if globals.retrain ~= "none" then
            local parameters, gradParameters = model:getParameters()
            local mod2 = torch.load(globals.retrain):double()
            local p2,gp2 = mod2:getParameters()
            parameters:copy(p2)
            gradParameters:copy(gp2)
        end
        -- model:cuda()
        parameters,gradParameters = model:getParameters()
        collectgarbage()
    end
    -- epoch tracker
    epoch = epoch or 1
    -- local vars
    local batchSize = globals.batchSize
    local epochSize = globals.epochSize
    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize .. ']')
    local tMSE = 0
    for t = 1,epochSize,batchSize do
        -- disp progress
        if globals.progressBar then xlua.progress(t, epochSize) end
        -- create mini batch
        local inputs, targets = load_data.getBatch(batchSize)
        -- inputs = inputs:cuda()
        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- get new parameters
            if x ~= parameters then
                parameters:copy(x)
            end
            -- reset gradients
            gradParameters:zero()
            -- f is the average of all criterions
            local f = 0;
            -- evaluate function for complete mini batch
            -- estimate f
            for i=1,batchSize do
                local output = model:updateOutput(inputs[i], targets[i])
                f = f + output

                -- estimate df/dW
                local df_do = criterion:backward(output, targets[i])
                model:backward(inputs[i], df_do)
                confusion:add(output, targets[i])
            end
            gradParameters:div(batchSize)
            -- fgradParameters:mul(#branch)
            f = f/batchSize
            -- return f and df/dX
            return f,gradParameters
        end
        -- optimize on current mini-batch
        optim.sgd(feval, parameters, optimState)
    end
    -- time taken
    local rMSE = math.sqrt(tMSE / (epochSize))
end


trainer = {}
trainer.train = train
return trainer
