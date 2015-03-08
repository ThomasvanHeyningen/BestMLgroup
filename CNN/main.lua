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

unlabeledSet = true
if unlabeledSet then
    local autoEncoder = unsup.AutoEncoder(model, decoder)
    autoEncoder.beta   = 6.0
    autoEncoder = trainer.train(autoEncoder)
    model = autoEncoder.encoder
    decoder = autoEncoder.decoder
end

if trainSet then
    model = net.addClassifier(model)
    model = trainer.train(model, trainSet)
end

local today = os.date('_%d_%m_%y')
torch.save(globals.clsfDir .. 'feature_extractor' .. today .. '.dat', model)
torch.save(globals.clsfDir .. 'decoder'           .. today .. '.dat', decoder)
