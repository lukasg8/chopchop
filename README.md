# chopchop

Guide to use:
1. At the bottom of OCRtoExcel.py, you will find 5 variables.
2. Input the variables which suit your application.
- Each variable has a description for you to understand what it does
3. Run the program


Potential Improvements:
1. When finding bounding boxes for digits, perhaps write a algorithm that identifies multiple boxes with the correct dimensions and uses it to
set the correct w and h of the boundingRec

2. add some data checking
i.e. if num outputted is over 1000, only take the last three digits
when you input it into the excel file, make sure the cell with the modified 
data is flagged (maybe highlight orange)