require 'nn'

local imgSize = {50,50}
local dataDir = '/home/harmen/Programming/neural_net/data/'
local mainDir = '/home/harmen/Programming/neural_net/'
local clsfDir = mainDir .. 'classifier/'
local saveDir = clsfDir

-- local netConfigFile = mainDir .. 'net_config.ini'
lblfile = dataDir .. 'lblfile.csv'

local currentNet   = 'net1'
local pruneData    = true
local pruneNr      = 3
local retrain      = 'none'
local batchSize    = 1
local progressBar  = true
local epochSize    = 798
local maxIter      = 2000
local learningrate = 2e-3
local decay        = 1e-5
local momentum     = 0.0
local statinterval = 100

local function isImgFile(str)
    return string.find(str, '.jpg') or string.find(str, '.JPG')
end

globals = {}
globals.imgSize = imgSize
globals.dataDir = dataDir
globals.mainDir = mainDir
globals.clsfDir = clsfDir
globals.saveDir = saveDir
-- globals.netConfigFile = netConfigFile
globals.currentNet   = currentNet
globals.lblfile      = lblfile
globals.isImgFile    = isImgFile
globals.pruneData    = pruneData
globals.pruneNr      = pruneNr
globals.retrain      = retrain
globals.batchSize    = batchSize
globals.progressBar  = progressBar
globals.epochSize    = epochSize
globals.maxIter      = maxIter
globals.learningrate = learningrate
globals.decay        = decay
globals.momentum     = momentum
globals.statinterval = statinterval
return globals
