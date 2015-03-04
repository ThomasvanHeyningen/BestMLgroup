require 'nn'

local imgSize = {50,50}
local dataDir = '/home/harmen/Programming/BestMLgroup/data/train/'
local mainDir = '/home/harmen/Programming/BestMLgroup/CNN'
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
local epochSize    = 8000
local maxIter      = 2000
local learningrate = 2e-3
local decay        = 1e-5
local momentum     = 0.0
local statinterval = 500

local function isImgFile(str)
    return string.find(str, '.jpg') or string.find(str, '.JPG')
end
local function isDir(str)
    return not string.find(str, '%.')
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
globals.isDir        = isDir
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
