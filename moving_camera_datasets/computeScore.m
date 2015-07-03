function[precision, recall, accuracy] = computeScore(groundTruthFile, imageFile)

gt = imread(groundTruthFile);
subject = imread(imageFile);

tp = nnz(gt .* subject);
tn = nnz((1 - gt) .* (1 - subject));
fp = nnz((1 - gt) .* subject);
fn = nnz(gt .* (1 - subject));

precision = tp / (tp + fp);
recall = tp / (tp + fn);
accuracy = (tp + tn) / (tp + tn + fp + fn);

end
