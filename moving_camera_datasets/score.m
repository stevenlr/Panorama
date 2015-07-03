disp("=== people1 ===");
disp("\nframe 1");
[precision, recall, accuracy] = computeScore("ground_truth/people1_01.pgm", "examples/mad/people1/output_foreground_clean_0001.png")
disp("\nframe 10");
[precision, recall, accuracy] = computeScore("ground_truth/people1_10.pgm", "examples/mad/people1/output_foreground_clean_0010.png")
disp("\nframe 20");
[precision, recall, accuracy] = computeScore("ground_truth/people1_20.pgm", "examples/mad/people1/output_foreground_clean_0020.png")
disp("\nframe 30");
[precision, recall, accuracy] = computeScore("ground_truth/people1_30.pgm", "examples/mad/people1/output_foreground_clean_0030.png")
disp("\nframe 40");
[precision, recall, accuracy] = computeScore("ground_truth/people1_40.pgm", "examples/mad/people1/output_foreground_clean_0040.png")

disp("\n\n=== people2 ===");
disp("\nframe 1");
[precision, recall, accuracy] = computeScore("ground_truth/people2_01.pgm", "examples/mad/people2/output_foreground_clean_0001.png")
disp("\nframe 10");
[precision, recall, accuracy] = computeScore("ground_truth/people2_10.pgm", "examples/mad/people2/output_foreground_clean_0010.png")
disp("\nframe 20");
[precision, recall, accuracy] = computeScore("ground_truth/people2_20.pgm", "examples/mad/people2/output_foreground_clean_0020.png")
disp("\nframe 30");
[precision, recall, accuracy] = computeScore("ground_truth/people2_30.pgm", "examples/mad/people2/output_foreground_clean_0030.png")
