globals = require 'globals'
require 'lfs'
require 'image'
require 'nn'
require 'torch'

local dataDir = globals.dataDir
local imgW = globals.imgSize[1]
local imgH = globals.imgSize[2]
local lblDict  = {}
local fileList = {}

local function preload()
    if string.find(dataDir, 'train') then
        for dir in lfs.dir(dataDir) do
            if globals.isDir(dir) then
                for file in lfs.dir(dataDir .. dir) do
                    if globals.isImgFile(file) then
                        table.insert(fileList, file)
                        lblDict[file] = dir
                    end
                end
            end
        end
    else
        for file in lfs.dir(dataDir) do
            if globals.isImgFile(file) then
                table.insert(fileList, file)
                lblDict[file] = '' -- No labels
            end
        end
    end
end

local function getData(files, tensor)
    if tensor == nil then tensor = true end
    if tensor then
        data = torch.DoubleTensor(#files, 1, imgW, imgH)
    else
        data = {}
    end
    for i, file in ipairs(files) do
        v = image.load(dataDir .. lblDict[file] .. '/' .. file)
        v = torch.reshape(v, 1, imgW, imgH):transpose(2,3)
        data[i] = v
    end
    return data
end

local function getLbls(files)
    local lblList = {}
    for i, file in ipairs(files) do
        lblList[i] = lblDict[file]
    end
    return lblList
end

local function getBatch(batchSize, tensor, labeled)
    if #fileList < batchSize then
        preload()
    end

    if tensor == nil then tensor = true end
    fileList, batchList = table.splice(fileList, 1, batchSize)
    
    batch = getData(batchList, tensor)
    if labeled then
        lblList = getLbls(batchList)
        return batch, lblList
    else
        return batch
    end
end

local function getAll(tensor, labeled)
    if tensor  == nil then tensor  = true end
    if labeled == nil then labeled = true end
    return getBatch(#fileList, tensor, labeled)
end

function get(labeled)
    local labeled = labeled or false
    preload()
    return getBatch(#fileList, true, labeled)
end


load_data = {}
load_data.get = get
load_data.getBatch = getBatch
load_data.getAll = getAll
return load_data
