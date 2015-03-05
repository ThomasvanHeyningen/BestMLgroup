globals = require 'globals'
require 'lfs'
require 'image'
require 'nn'

local dataDir = globals.dataDir
local imgW = globals.imgSize[1]
local imgH = globals.imgSize[2]
local lblDict  = {}
local lblList  = {}
local fileList = {}
local labeled  = true

for dir in lfs.dir(dataDir) do
    if globals.isDir(dir) then
        for file in lfs.dir(dataDir .. dir) do
            if globals.isImgFile(file) then
                table.insert(fileList, {dir,file})
            end
        end
    end
end

local function preprocess(data)
    -- data = data:float() -- Doubles are expected
    -- traindata.channelNames = {'y','u','v'} -- Somehow breaks trainData
    local len = #data
    if type(len) ~= 'number' then 
        len = len[1]
    end
    return data
end

local function getData(files, tensor)
    if tensor == nil then tensor = true end
    if tensor then
        local data = torch.Tensor(#files, 3, imgW, imgH)
    else
        local data = {}
    end
    for i, file in ipairs(files) do
        file, lbl = table.splice(file,1)
        file = file[1]; lbl = lbl[1]
        v = image.load(dataDir .. lbl .. '/' .. file):transpose(2,3)
        if tensor then
            v = torch.reshape(v, 1, imgW, imgH, 1)
        else
            v = torch.reshape(v, 1, imgW, imgH)
        end
        data[i] = v
        lblList[i] = file
    end
    return data
end

local function getBatch(batchSize, tensor)
    if tensor == nil then tensor = true end
    fileList, batchList = table.splice(fileList, 1, batchSize)
    
    batch = getData(batchList, tensor)
    batch = preprocess(batch)
    if labeled then
        lblList,  batchlbls = table.splice(lblList,  1, batchSize)
        return batch, batchlbls
    else
        return batch, batch
    end
end

local function getAll(tensor)
    if tensor == nil then tensor = true end
    return getBatch(#fileList, tensor)
end

local function setLabeled(state)
    labeled = state
end

function get(labeled)
    labeled = labeled or false
    
    trainData = {}
    if labeled then
        getLabels(globals.lblfile)
    end
    trainData.data = getData(fileList)
    trainData.size = (#trainData.data)[1]
    
    -- Preprocess
    trainData.data = preprocess(trainData.data)
    if not labeled then
        trainData.labels = trainData.data
    else
        trainData.labels = torch.Tensor(lblList)
    end
    return trainData
end


load_data = {}
load_data.get = get
load_data.getBatch = getBatch
load_data.setLabeled = setLabeled
load_data.getAll = getAll
return load_data
