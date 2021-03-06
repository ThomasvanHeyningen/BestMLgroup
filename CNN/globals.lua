require 'nn'

local imgSize  = {50,50}
local dataDir  = '/home/harmen/Programming/data/'
local trainDir = dataDir .. 'train/'
local testDir  = dataDir .. 'test/'
local mainDir  = '/home/harmen/Programming/BestMLgroup/CNN/'
local clsfDir  = mainDir .. 'classifier/'
local saveDir  = clsfDir
local nclass   = 121

-- local netConfigFile = mainDir .. 'net_config.ini'
lblfile = dataDir .. 'lblfile.csv'

local currentNet   = 'net2'
local pruneData    = true
local pruneNr      = 3
local retrain      = 'none'
local batchSize    = 100
local progressBar  = true
local epochSize    = 30000
local maxIter      = 2000
local learningrate = 1e-5
local decay        = 1e-9
local momentum     = 0.5
local statinterval = 1

local function isImgFile(str)
    return string.find(str, '.jpg') or string.find(str, '.JPG')
end
local function isDir(str)
    return not string.find(str, '%.')
end

globals = {}
globals.imgSize  = imgSize
globals.dataDir  = dataDir
globals.trainDir = trainDir
globals.testDir  = testDir
globals.mainDir  = mainDir
globals.clsfDir  = clsfDir
globals.saveDir  = saveDir
globals.nclass   = nclass
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
