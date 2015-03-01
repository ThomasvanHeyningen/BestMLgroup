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

for file in lfs.dir(dataDir) do
    if globals.isImgFile(file) then
        table.insert(fileList, file)
    end
end

local openfile = io.open(globals.lblfile)
for row in openfile:lines() do
    pair = string.split(row, ',')
    key = pair[1] -- Filename
    val = pair[2] -- Label
    lblDict[key] = val
end

local function preprocess(data)
    -- data = data:float() -- Doubles are expected
    -- traindata.channelNames = {'y','u','v'} -- Somehow breaks trainData
    local len = #data
    if type(len) ~= 'number' then 
        len = len[1]
    end
    for i = 1, len do
	data[i] = image.rgb2yuv(data[i])
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
        v = image.load(dataDir .. file):transpose(2,3)
        if tensor then
            v = torch.reshape(v, 3, imgW, imgH, 1)
        else
            v = torch.reshape(v, 3, imgW, imgH)
        end
        data[i] = v
    end
    -- List labels such that lblList[i] corresponds to data[i]
    for i, file in ipairs(files) do
        lblList[i] = lblDict[file]
    end
    return data
end

local function getBatch(batchSize, tensor)
    if tensor == nil then tensor = true end
    fileList, batchList = table.splice(fileList, 1, batchSize)
    lblList,  batchlbls = table.splice(lblList,  1, batchSize)
    
    if tensor then
        batch = getData(batchList, true)
    else
        batch = getData(batchList, false)
    end
    batch = preprocess(batch)
    if labeled then
        lbls = torch.Tensor(lblList)
        return batch, batchlbls
    else
        return batch, batch
    end
    
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
return load_data
