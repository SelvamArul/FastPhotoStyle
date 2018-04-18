require 'torch_refinenet_ext'
-- require 'image_augmentation'
require 'paths'

local InferenceLoader = torch.class('InferenceLoader')



function InferenceLoader:__init(opt, old_classes, load_stride)
    print('inference loader init')
	assert(opt.dataset)
	self.dataset_path = opt.dataset
	self.image_size = opt.image_size
	self.load_stride = load_stride or 1

    print('in inference loader dataset is ', self.dataset_path)

	-- self.annotated_frames_path = paths.concat(self.dataset_path, 'annotated')
    -- print(self.annotated_frames_path)

    -- self.images_path=paths.concat(self.dataset_path, 'cam_left/')
    self.images_path=self.dataset_path

	-- Load list of frames for train and test split
	self.frames = {}
	self:loadDirectory(self.frames, self.images_path )
	print(string.format('Found %d frames', #self.frames))

	-- Load list of classes
	local classes = {}
	local current_idx = 1


	local f = assert(io.open(paths.concat(self.dataset_path, "../classes.txt")))

	while true do
		local line = f:read()
		if line == nil then
			break
		end

		if line:len() ~= 0 and line:sub(1,1) ~= '#' then
			-- Do we already know this one?
			if not classes[line] then
				classes[line] = current_idx
				current_idx = current_idx + 1
			end
		end
	end


	self.classToID = classes
	self.idToClass = {}
	for k,v in pairs(classes) do
		self.idToClass[v] = k
	end

	for k,v in pairs(self.idToClass) do
		print(k, v)
	end

	print('Number of classes:', #self.idToClass)

	self.resnet_meanstd = {
		mean = { 0.485, 0.456, 0.406 },
		std = { 0.229, 0.224, 0.225 },
	}

    -- print('starting instatiating frame loader')
	-- self.loader = torch_refinenet_ext.FrameLoader(
		-- self.dataset_path,
		-- self.classToID,
		-- self.image_size
	-- )
    -- print('finished instatiating frame loader')

	-- Queue first images
	print('Starting prefetch for train and test data')

	-- self.prefetch_depth = 8
	-- for i=1,self.prefetch_depth do
	-- 	local idx = ((self.load_stride*i-1) % #self.frames) + 1
    --     print('queuing in the frame', self.frames[idx])
	-- 	self.loader:queueFrame(self.frames[idx])
    --
	-- end
end

function InferenceLoader:loadDirectory(frames_out, path)

    print('loading from ', path)
    --get all the files in the folder
    for file in paths.iterfiles(path) do
        table.insert(frames_out, paths.concat(path, file))
    end

end


function InferenceLoader:numFrames()
    return #self.frames
end

-- function InferenceLoader:getFrame(idx, loader_opt)
--
-- 	local frames
-- 	local loader
-- 	local stride
--
-- 	frames = self.frames
-- 	loader = self.loader
-- 	stride = self.load_stride
--
-- 	local frame = loader:getFrame()
--
-- 	-- Batch the image
-- 	frame.image = frame.image:view(1, frame.image:size(1), frame.image:size(2), frame.image:size(3))
-- 	frame.label = frame.label:view(1, frame.label:size(1), frame.label:size(2))
--     frame.label:add(1) --we add 1 because the labels in lua start from one but int he opencv label img they start from 0
--     -- min = torch.min(frame.label)
--     -- max = torch.max(frame.label)
--     -- print('min is ',min)
--     -- print('max is ',max)
--
-- 	frame.rawImg = frame.image[1]:clone()
--
-- 	-- centering for ResNet
-- 	for i=1,3 do
-- 		frame.image[1][i]:add(-self.resnet_meanstd.mean[i])
-- 		frame.image[1][i]:div(self.resnet_meanstd.std[i])
-- 	end
--
-- 	-- Start preloader for next frame
-- 	-- FIXME This is ugly and assumes that we call this function with monotone idx sequences.
-- 	do
-- 		local newIdx = ((idx + stride*self.prefetch_depth - 1) % #frames) + 1
-- 		loader:queueFrame(frames[newIdx])
-- 	end
--
-- 	return frame
-- end

function InferenceLoader:numClasses()
	return #self.idToClass
end
