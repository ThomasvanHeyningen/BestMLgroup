require 'lfs'
require 'image'
require 'globals'
require 'nn'

local rawDir = '/home/harmen/PycharmProjects/image_feature_selection/data/Barcelona 2008/'
local data = {}
data[-1] = {} -- Only for testing
data[0] = {}
data[1] = {}

local function isDir(file)
    return string.find(file, '.', 1, true) == nil
end

local function parseImage(file, depth)
    local img = image.load(file)
    img = image.scale(img, 100, 100)
    data[depth][table.getn(data[depth]) + 1] = img
end

local function parseDir(dirname, depth)
    print('Searching in ' .. dirname)
    for file in lfs.dir(dirname) do
        if globals.isImgFile(file) then
            parseImage(dirname .. file, depth)
        elseif isDir(file) then
            parseDir(dirname .. file .. '/', depth+1)
        end
    end
end

parseDir(rawDir, -1)
torch.save('/home/harmen/Programming/neural_net/data/ReSnap4K.t7')
