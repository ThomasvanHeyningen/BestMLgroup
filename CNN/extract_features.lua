require 'lfs'
require 'globals'
require 'load_data'
require 'nn'
require 'unsup'
require 'math'
require 'csvigo'

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
batchSize = 2000
maxImgs = 30300

openfile = io.open('features.csv' , 'w') -- Remove previous features
local n = 1764
for b = 1,maxImgs, batchSize do
    print('Parsing imgs ' .. b .. ' to ' .. (b + batchSize))
    local imgs, lbls = load_data.getBatch(batchSize, false)

    for i = 1,#lbls do
        local feature = extractor:forward(imgs[i])
        feature = torch.reshape(feature, n)
        feature = torch.totable(feature)
        feature = table.concat(feature, ',')
        feature = lbls[i] .. ',' .. feature
        openfile:write(feature)
        openfile:write('\n')
    end
end
openfile:close()
