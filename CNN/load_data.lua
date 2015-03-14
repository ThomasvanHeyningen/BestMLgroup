globals = require 'globals'
require 'lfs'
require 'image'
require 'nn'
require 'torch'
require 'csvigo'

local dataDir = nil
local trainDir = globals.trainDir
local testDir = globals.testDir
local imgW = globals.imgSize[1]
local imgH = globals.imgSize[2]
local lblDict  = {}
local fileList = {}
local nclasses = globals.nclass
local lbl2num  = {}

local function setDir(which)
    if which == 'train' then
        dataDir = trainDir
    else
        dataDir = testDir
    end
    fileList = {}
end

local function getLbl2Num()
    lst = csvigo.load('classlist.csv')
    dct = {}
    for v,i in pairs(lst) do -- v = name, i = number
        dct[v] = tonumber(i[1])
    end
    return dct
end

local function preload()
    if string.find(dataDir, 'train') then
        lbl2num = getLbl2Num()
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
    print(#fileList)
end

local function getData(files, tensor)
    if tensor == nil then tensor = true end
    if tensor then
        data = torch.DoubleTensor(#files, 1, imgW, imgH)
    else
        data = {}
    end
    for i, file in ipairs(files) do
        lbl = lblDict[file]
        if string.sub(lbl, -1) ~= '/' and #lbl > 1 then lbl = lbl .. '/' end
        v = image.load(dataDir .. lbl ..  file)
        v = torch.reshape(v, 1, imgW, imgH):transpose(2,3)
        data[i] = v
    end
    return data
end

--[[
local function oneHot(p,n)
    vec = torch.zeros(n)
    vec[p] = 1
    return vec
end
--]]

local function getLbls(names)
    local  lblList = {}
    for i, name in ipairs(names) do
        lblList[i]  = lbl2num[lblDict[name]]
    end
    return lblList
end

local function getBatch(batchSize, tensor, labeled)
    if #fileList < batchSize then
        preload(batchSize)
    end

    if tensor == nil then tensor = true end
    fileList, batchList = table.splice(fileList, 1, batchSize)
    
    batch = getData(batchList, tensor)
    lblList = getLbls(batchList)
    if labeled then
        return batch, lblList, batchList
    else
        return batch, fileList
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
load_data.setDir = setDir
load_data.get = get
load_data.getBatch = getBatch
load_data.getAll = getAll
return load_data
