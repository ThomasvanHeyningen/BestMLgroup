require 'torch'
require 'unsup'
require 'globals'
require 'load_data'
require 'trainer'
net = require(globals.currentNet) -- Rename because the nets have different names
require 'nn'

--unlabeledSet = load_data.get(false)
--local trainSet = load_data.get(true)
--local testSet = load_data.get(true)

local model = net.stageOne()
local decoder = net.stageTwo()

local unlabeledSet = true
if unlabeledSet then
    load_data.setDir('test')
    local autoEncoder = unsup.AutoEncoder(model, decoder)
    autoEncoder.beta   = 6.0
    autoEncoder = trainer.train(autoEncoder, true)
    model = autoEncoder.encoder
    decoder = autoEncoder.decoder
end

local trainSet = true
if trainSet then
    load_data.setDir('train')
    model = net.addClassifier(model)
    model = trainer.train(model, false)
end

local today = os.date('_%d_%m_%y')
torch.save(globals.clsfDir .. 'feature_extractor' .. today .. '.dat', model)
torch.save(globals.clsfDir .. 'decoder'           .. today .. '.dat', decoder)
