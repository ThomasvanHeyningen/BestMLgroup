require 'lfs'
require 'globals'
require 'load_data'
require 'nn'
require 'unsup'
require 'math'
require 'torch'

dir = globals.clsfDir
map = globals.map

local function latestModel(model_name)
    comp = function(a,b) -- Sorts year>month>day with biggest first 
        if tonumber(a[3]) == tonumber(b[3]) then
            if tonumber(a[2]) == tonumber(b[2]) then
                return tonumber(a[1]) > tonumber(b[1])
            else
                return tonumber(a[2]) > tonumber(b[2])
            end
        else
            return tonumber(a[3]) > tonumber(b[3])
        end
    end
    dates = {}
    for filename in lfs.dir(dir) do
        if string.find(filename, model_name) then
            date = string.split(string.sub(filename, #model_name, #model_name+8), '_')
            table.insert(dates, date)
        end
    end
    table.sort(dates, comp)
    return model_name .. table.concat(dates[1], '_') .. '.dat'
end

extractor = torch.load(dir .. (arg[2] or latestModel('feature_extractor_')))
extractor:evaluate()
decoder   = torch.load(dir .. (arg[3] or latestModel('decoder_')))
decoder:evaluate()

i = torch.random(5)
img = load_data.getBatch(5, true, false)[i]
image.display(img)
--[[
features = extractor:get(1):forward(img)
features = extractor:get(2):forward(features)
image.display(features[i])

features = extractor:get(3):forward(features)
features = extractor:get(4):forward(features)
features = extractor:get(5):forward(features)
features = extractor:get(6):forward(features)
image.display(features[i])

features = extractor:get(7):forward(features)
features = extractor:get(8):forward(features)
reconstruction = decoder:forward(features)
image.display(image.yuv2rgb(reconstruction))
--]]

image.display{
    image = image.toDisplayTensor(
        {input=extractor.modules[1].weight, padding=2,
         nrow=4, symmetric=true}
    ),
    zoom=10
}

image.display{
    image = image.toDisplayTensor(
        {input=extractor.modules[5].weight, padding=2,
         nrow=8, symmetric=true}
    ),
    zoom=5
}

image.display{
    image = image.toDisplayTensor(
        {input=extractor.modules[9].weight, padding=2,
         nrow=8, symmetric=true}
    ),
    zoom=5
}

--[[
AA = unsup.AutoEncoder(extractor, decoder)
AA.beta = 6.0
err = AA:updateOutput(img, img)
print(err)
--]]

