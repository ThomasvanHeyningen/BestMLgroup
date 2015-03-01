require 'torch'
require 'globals'
require 'load_data'

date = arg[1]
dir  = arg[2]

classifier_name = 'classifier_' .. date .. '.dat'
classifier = torch.load(globals.clsfDir + classifier_name)
data = load_data.get(false).data

features = {}
for i = 1,(#data)[1] do
    features[i] = classifier:forward(data[i][{{2,3}}])
end

torch.write('features.dat', features)
