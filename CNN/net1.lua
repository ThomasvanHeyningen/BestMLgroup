require 'nn'
require 'image'
require 'globals'

-- Parameters
local nclasses  = 1
local nchannels = {3, 4, 4, 32, nclasses}
local fanIn     = {1} -- #Maps in previous layer to connect to
local kernSize  = {9} -- Eye width and height
local P         = 2   -- Pooling agression, 1 = mean, inf = max
local pool      = {{w = 2, h = 2, dw = 2, dh = 2}}
local normKern  = image.gaussian1D(7)
local imgWidth  = globals.imgSize[1]
local imgHeight = globals.imgSize[2]

local dims = {} -- Shape of every map in layer
dims[1]    = {imgWidth, imgHeight}
dims[2]    = {imgWidth  - kernSize[1] + 1, 
              imgHeight - kernSize[1] + 1}
dims[3]    = {dims[2][1] / 2, dims[2][2] / 2}


function stageOne()
    model = nn.Sequential()
    -- Layer 1 is the image
    -- One set Deep NN
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nchannels[1], nchannels[2], fanIn[1]),
	      kernSize[1], kernSize[1])) -- Layer 2
    model:add(nn.Tanh())
    model:add(nn.SpatialLPPooling(nchannels[2], P,
	      pool[1]['w'], pool[1]['h'], pool[1]['dw'], pool[1]['dh'])) -- Layer 3
    model:add(nn.SpatialSubtractiveNormalization(nchannels[2], normKern))
    return model
end

function stageTwo()
    -- Add a linear layer as decoder
    decoder = nn.Sequential()
    
    -- Flatten Layer 3
    local length = nchannels[3] * dims[3][1] * dims[3][2]
    decoder:add(nn.Reshape(length))
    decoder:add(nn.Linear(length, nchannels[1] * imgWidth * imgHeight)) 
    decoder:add(nn.Reshape(nchannels[1], imgWidth, imgHeight)) -- Output is an image
    return decoder
end

function addClassifier(model)
    -- Append a two-layer NN
    
    -- Flatten Layer 3
    local length = nchannels[3] * dims[3][1] * dims[3][2]
    model:add(nn.Reshape(length))
    model:add(nn.Linear(length, nchannels[3])) -- Layer 4
    model:add(nn.Tanh())
    model:add(nn.Linear(nchannels[3], nchannels[5])) -- Layer 5, output is p(class)
    return model
end


net1 = {}
net1.stageOne = stageOne
net1.stageTwo = stageTwo
net1.addClassifier = addClassifier
return net1
