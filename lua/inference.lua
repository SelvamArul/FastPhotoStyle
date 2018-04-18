package.path = './lua/?.lua;' .. package.path

require 'RefineNetModel'
require 'cunn'
require 'cudnn'
require 'image'
require 'InferenceLoader'
require 'math'

--load label_to_meta_label
dofile('label_map.lua')

local visdom = require 'visdom'
local visdom_ext = require 'visdom_ext'
local vis_utils = require 'vis_utils'

-- CUDA_VISIBLE_DEVICES=3 th lua/inference.lua -dataset /home/local/stud/rosu/data/janis_data/cam_left -output /home/local/stud/rosu/data/janis_data/cam_left_segmentation -checkpoint /home/local/stud/rosu/data/checkpoints/e_2_iter_18087_crash_7/checkpoint.t7
torch.setdefaulttensortype('torch.FloatTensor')

local model
local resnet_meanstd = {
	mean = { 0.485, 0.456, 0.406 },
	std = { 0.229, 0.224, 0.225 },
}
local IMAGE_SIZE = 1024

local opt

do
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Inference on a Segmentation model.')
  cmd:text()
  cmd:text('Options')

  -- Model settings
  cmd:option('-checkpoint', '', 'Load model checkpoint')

  -- Data input settings
  cmd:option('-dataset', '/home/local/arc', 'Path to the dataset')
  cmd:option('-image_size', 1024, "Image size")

  cmd:text()
  opt = cmd:parse(arg)
end

--visdom client
local plot = visdom{server = 'http://localhost', port = 5000, ipv6=false, env=opt.checkpoint_path}
plot:close{win=nil}
windows = {}



print('Loading checkpoint', opt.checkpoint)
local model = torch.load(opt.checkpoint).model
-- local loader = DirectLoader(opt, model.classToID)
print('Loading dataset', opt.dataset)
local loader = InferenceLoader(opt, model.classToID)

opt.num_classes = loader:numClasses()
print("Number of classes:", opt.num_classes)

model:evaluate()


--read img from loader.frames and run them through the network
print("num classes", opt.num_classes)
local images = {}

local num_images = loader:numFrames()
for i=1,num_images do
	-- Grab a batch of data and convert it to the right dtype

	local img
	print('inference on img ', loader.frames[i], string.format("%d/%d", i, num_images))
	img=image.load(loader.frames[i],3, 'float')
	local raw_img=img;

	--scale it to fit inside the IMAGE_SIZE
	scale = IMAGE_SIZE / math.max(img:size(2), img:size(3));
	img = image.scale(img, IMAGE_SIZE)


	-- centering for ResNet
	for i=1,3 do
		img[i]:add(-resnet_meanstd.mean[i])
		img[i]:div(resnet_meanstd.std[i])
	end
	--get size of output
	local prediction_width = math.floor((img:size(3) - 1)/2 + 1)
	prediction_width = math.floor((prediction_width - 1)/2 + 1)
	local prediction_height = math.floor((img:size(2) - 1)/2 + 1)
	prediction_height = math.floor((prediction_height - 1)/2 + 1)

	--forward
	img = img:view(1,3,img:size(2),img:size(3))
	local prediction = model:forward(img:cuda())
	-- print('size of prediction is ', prediction:size())
	local predict_img = prediction:view(prediction_height, prediction_width,  opt.num_classes)
	-- print('size of predict_img is ', predict_img:size())
	-- print('prediction aspect ratio is ', predict_img:size(2)/predict_img:size(1))

	local prob, classes = torch.max(predict_img, 3)
	classes = classes:view(prediction_height, prediction_width)  --get them as 2D tensors and not 3D
	prob = prob:view(prediction_height, prediction_width)
	raw_img = image.scale(raw_img, math.max(prediction_height, prediction_width))


	--write the classes and probabilities to images
	classes = classes:view(1, prediction_height, prediction_width)
	prob = prob:view(1, prediction_height, prediction_width)

	local file_name = paths.basename(loader.frames[i], '.jpg')
	local output_path = paths.dirname(loader.frames[i])
	-- print('file_name is', file_name)
	local file_name_classes=file_name .. '_labels' .. '.png'
	local file_name_probs=file_name .. '_probs' .. '.png'

	local file_name_meta_classes = file_name_classes:clone()
	for _i=1,66 do
		file_name_meta_classes [file_name_classes == _i] = label_to_meta_label[_i]
	end

	local out_path_classes=paths.concat(output_path, file_name_classes)
	local out_path_probs=paths.concat(output_path, file_name_probs)

	--the save function requires the tensors to be in range 0-1
	classes = torch.div(classes:float(), 255)
	prob=torch.exp(prob) -- the last layer of LogSoftMax outputs log probability so to get probability we do exp

	image.save(out_path_classes, classes)
	image.save(out_path_probs, prob)
end


collectgarbage()
collectgarbage()
