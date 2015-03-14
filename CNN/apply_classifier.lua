require 'torch'
require 'globals'
require 'load_data'
require 'nn'

local function latestDate(model_name)
    dir = globals.clsfDir
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
    return table.concat(dates[1], '_')
end


date = arg[1] or latestDate('feature_extractor_')
print(date)

model_name = 'feature_extractor_' .. date .. '.dat'
model_path = globals.clsfDir .. model_name
model = torch.load(model_path)
extractor = nn.Sequential()
for i = 1,12 do
    extractor:add(model:get(i))
end
classifier = nn.Sequential()
for i = 13,15 do
    classifier:add(model:get(i))
end

load_data.setDir('test')
n = 130276
for i = 1,n,2000 do
    data = load_data.getBatch(2000, true, false)

    features = {}
    for i = 1,(#data)[1] do
        feature = extractor:forward(data[i][{{2,3}}])
        pred = classifier:forward(feature)
        features.cat(torch.cat(feature, pred))
    end
end
torch.write('features.dat', features)
