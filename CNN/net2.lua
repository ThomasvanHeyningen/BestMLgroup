require 'nn'
require 'image'
require 'globals'

-- Parameters
local nclasses  = globals.nclass
local nchannels = {1, 8, 8, 32, 32, 16, 16, 1, 1, nclasses} -- [1] = 3
local fanIn     = {1, 6, 6}
local kernSize  = {3, 3, 5}
local P         = 2
local pool      = {{w = 2, h = 2, dw = 2, dh = 2},
                   {w = 2, h = 2, dw = 2, dh = 2},
                   {w = 2, h = 2, dw = 1, dh = 1}}
local normKern  = image.gaussian1D(7)
local imgWidth  = globals.imgSize[1]
local imgHeight = globals.imgSize[2]

local dims = {}
dims[1]    = {imgWidth, imgHeight}
dims[2]    = {dims[1][1] - kernSize[1] + 1,
              dims[1][2] - kernSize[1] + 1}
dims[3]    = {dims[2][1] / 2,
              dims[2][2] / 2}
dims[4]    = {dims[3][1] - kernSize[2] + 1,
              dims[3][2] - kernSize[2] + 1}
dims[5]    = {dims[4][1] / 2,
              dims[4][2] / 2}
dims[6]    = {dims[5][1] - kernSize[3] + 1,
              dims[5][2] - kernSize[3] + 1}
dims[7]    = {dims[6][1] - 1,
              dims[6][2] - 1}
dims[8]    = {dims[7][1] * dims[7][2] * nchannels[7]}
dims[9]    = {256}


function stageOne()
    model = nn.Sequential()
    model:add(nn.SpatialConvolutionMap(nn.tables.random(nchannels[1], nchannels[2], fanIn[1]),
                                       kernSize[1], kernSize[1]))
    model:add(nn.ReLU())
    model:add(nn.SpatialLPPooling(nchannels[2], P, pool[1]['w'], pool[1]['h'], pool[1]['dw'], pool[1]['dh']))
    model:add(nn.SpatialSubtractiveNormalization(nchannels[2], normKern))

    model:add(nn.SpatialConvolutionMap(nn.tables.random(nchannels[3], nchannels[4], fanIn[2]),
                                       kernSize[2], kernSize[2]))
    model:add(nn.ReLU())
    model:add(nn.SpatialLPPooling(nchannels[4], P, pool[2]['w'], pool[2]['h'], pool[2]['dw'], pool[2]['dh']))
    model:add(nn.SpatialSubtractiveNormalization(nchannels[4], normKern))

    model:add(nn.SpatialConvolutionMap(nn.tables.random(nchannels[5], nchannels[6], fanIn[3]),
                                       kernSize[3], kernSize[3]))
    model:add(nn.ReLU())
    model:add(nn.SpatialLPPooling(nchannels[6], P, pool[3]['w'], pool[3]['h'], pool[3]['dw'], pool[3]['dh']))
    model:add(nn.SpatialSubtractiveNormalization(nchannels[6], normKern))

    return model
end

function stageTwo()
    decoder = nn.Sequential()

    decoder:add(nn.Reshape(dims[8][1]))
    decoder:add(nn.Linear(dims[8][1], dims[9][1]))
    decoder:add(nn.Linear(dims[9][1], nchannels[1] * imgWidth * imgHeight))
    decoder:add(nn.Reshape(nchannels[1], imgWidth, imgHeight))

    return decoder
end

function addClassifier(model)
    model:add(nn.Reshape(dims[8][1]))
    model:add(nn.Linear(dims[8][1], dims[9][1]))
    model:add(nn.Linear(dims[9][1], nclasses))
    
    return model
end

net2 = {}
net2.stageOne = stageOne
net2.stageTwo = stageTwo
net2.addClassifier = addClassifier
return net2
