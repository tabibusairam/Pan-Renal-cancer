function segmentedImage = hmt(I, minSize, verbose)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% hmt: this function implements the segmentation method proposed in the
% reference below. The algorithm was purposed to segment the nuclei inside
% H&E histology images.
%
% Input:
%       I - color image.
% minSize - (optional) minimum region size in pixels (default = 50).
% verbose - (optional) display results (default = false).
%
% Reference:
% [1] H. Ahmady Phoulady, D. B. Goldgof, L. O. Hall, and P. R. Mouton.,
% Nucleus segmentation in histology images with hierarchical multilevel
% thresholding, In SPIE Medical Imaging Symposium: Digital Pathology, pages
% 979111–979116. International Society for Optics and Photonics, 2016.
%
% @inproceedings{phoulady2016hmt,
% 	title={Nucleus segmentation in histology images with hierarchical
%   multilevel thresholding},
% 	author={Ahmady Phoulady, Hady and Goldgof, Dmitry B and Hall, Lawrence
%   O and Mouton, Peter R},
% 	booktitle={SPIE Medical Imaging Symposium: Digital Pathology},
% 	pages={979111--979116},
% 	year={2016},
% 	organization={International Society for Optics and Photonics}
% }
%
% Note: The included image is taken from a dataset provided by Wienert et
% al. The dataset can be downloaded from the publication webpage:
% http://www.nature.com/articles/srep00503
%
% Example:
% 	I = imread('sample.jpg');
% 	segmentedImage = hmt(I, [], true);
%
%
% Copyright (c) 2016, Hady Ahmady Phoulady
% Department of Computer Science and Engineering,
% University of South Florida, Tampa, FL.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The minimum size allowed for regions. It can be changed based on image
if (~exist('minSize', 'var') || isempty(minSize))
    minSize = 50;
end

% Should we overlay the region boundaries and display the result?
if (~exist('verbose', 'var'))
    verbose = false;
end


% Perform color deconvolution and extract hematoxylin channel
deconv = Deconvolve(I);
deconv = deconv(:, :, 1);
devonv = (-deconv - min(-deconv(:))) ./ (max(-deconv(:)) - min(-deconv(:)));
bwI = uint8(255 * devonv);

bwI = imcomplement(bwI);
openByReconstruction = imreconstruct(imerode(bwI, strel('disk', 3, 0)), bwI);
bwI = imcomplement(openByReconstruction);


segmentedImage = false(size(bwI));

% Calculating the thresholds by performing multilevel thresholding
maxLevel = 10;
levels = zeros(2 * maxLevel, 1);
for l = 1: maxLevel
    multiLevels = multithresh(bwI, l);
    levels(l) = multiLevels(1);
end
levels = sort(unique(levels), 'descend');

% Binarizing the image according to thresholds and splitting the regions
% hierarchically
for k = 1: length(levels)
    cells = bwI <= levels(k);
    cells = imfill(cells, 'holes');
    cells = imopen(cells, strel('disk', 3, 0));
    cells = bwareaopen(cells, minSize, 4);
    cellSize = nnz(cells);
    if (cellSize < 2 * minSize)
        break
    end

    s = regionprops(segmentedImage, 'PixelIdxList');
    removeRegions = false(length(s), 1);
    newRegions = bwlabel(cells, 4);
    for l = 1: length(s)
        if length(unique(newRegions(s(l).PixelIdxList))) > 2
            removeRegions(l) = true;
        end
    end
    segmentedImage(cat(1, s(removeRegions).PixelIdxList)) = false;
    segmentedImage = cells | segmentedImage;
end

% Removing boundary regions
imageBoundary = false(size(bwI));
imageBoundary(1, :) = true;
imageBoundary(end, :) = true;
imageBoundary(:, 1) = true;
imageBoundary(:, end) = true;
segmentedImage(imreconstruct(imageBoundary, segmentedImage)) = false;

% Expanding the region boundaries (except when it makes two regions merge)
dilatedCellSeg = imdilate(segmentedImage, strel('diamond', 1));
[regionsLabel, ~] = bwlabel(segmentedImage, 4);
[dilatedRegionsLabel, numOfDilatedRegions] = bwlabel(dilatedCellSeg, 4);
for l = 1: numOfDilatedRegions
    if (length(unique(regionsLabel(dilatedRegionsLabel == l))) >= 3)
        dilatedCellSeg(dilatedRegionsLabel == l) = segmentedImage(dilatedRegionsLabel == l);
    end
end
segmentedImage = dilatedCellSeg;

% Creating and displaying the result if verbose == true
if (verbose)
    regionBoundaries = find((bwdist(segmentedImage) <= 1 & ~segmentedImage)| bwmorph(segmentedImage, 'remove'));
    result = I;
    result(regionBoundaries) = 0;
    result(size(I, 1) * size(I, 2) + regionBoundaries) = 255;
    result(2 * size(I, 1) * size(I, 2) + regionBoundaries) = 0;
    figure('Name', 'Overlaid boundaries on original image'), imshow(result)
end