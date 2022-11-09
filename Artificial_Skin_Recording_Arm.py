import numpy as np
#import seaborn as sns; sns.set_theme()
#import matplotlib
import matplotlib.pyplot as plt
import serial 
import time
from timeit import default_timer as timer
import csv
from tkinter import *
import sys
import statistics
import xlsxwriter
import winsound
import keyboard

#######################PLEASE ENTER BEFORE USE#######################################
#init tactile array variables
num_sensors=80 #please enter the number of sensors in the array. Make sure to also enter the same number in the arduino array
time_step=0.200 #0.0500 (=50ms) #time interval in seconds to read data from the ARDUINO(values has to correspond with the one in the ARDUINO code)
time_recording=10 #recording time in seconds
num_rows=3 #For visualization only: Number of array rows
num_columns=1 #For visualization only: Number of array columns

input_type='test'#['VP', 'VP','VP','VP','VP','VP','VP','VP','VP'] #type of tactile imput per event -> 'VP'=Vertical Pressure
num_measurements=int(time_recording/time_step) #100 #number of expected measurements. If this number is exceeded, the data are saved and the program is closed

I2C_address=np.array([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,0x0F,77,78,79,0x0A,0x0B,0x0C,0x0D,0x0E])
###Other variables for Recording - No USER input required
raw_data=[[None for c in range(num_sensors+3)] for r in range(num_measurements)] #array for raw tactile data
lin_data= [[None for c in range(num_sensors+3)] for r in range(num_measurements)] #array for linearized tactile data
coeff_ref=[0]*2


label_csv=['Time']
sensor_ids=[]
for z in  range(1,num_sensors+1):
    sensor_ids.append(z)

reading_count=[0,0,0] #the first index contains the number of recorded data packages at the current iteration/time; the second index contains the number of recorded data sets that got saved already (during a previous intermediate saving action), the third number is the number of recorded data sets of theprevious time step
######## INIT serial communication variables################################
ser = serial.Serial('COM3', 57600)
ser.flushInput()
ser_input=[]
time.sleep(0.75)
pressure_label=[]
first_run = True
calibration=False
transfer_received=False


######### INIT data recording variables
array_plot=np.zeros((num_rows,num_columns)) #intialize pressure array with zeros, First: Number of array ROWS, Second: Number of array COLUMNS -> FOR VISUAL:ZATION ONLY
array_pressure=np.zeros((num_rows*num_columns)) #array to read and temporarily store pressure values

#HEATMAP PLOT INITIALIZATION###############
'''sensorX =[] #array or plotting the axis labels
sensorY =[] #array or plotting the axis labels
for p in range(num_columns):
    sensorX.append(p) #pseudo x-sensor coordinates for plotting
for t in range(num_rows):
    sensorY.append(t) #pseudo y-sensor coordinates for plotting

#enable interactive plotting mode (required for continously updating the plot with serial data)   
plt.ion()

fig, ax = plt.subplots()
# The subplot colors do not change after the first time
# if initialized with an empty matrix
im = ax.imshow(array_plot,cmap='jet', vmin=0.0, vmax=100.0)

# Major ticks every 1x, minor ticks every 0.5
major_ticksX = np.arange(0, num_columns,1)
minor_ticksX = np.arange(0, num_columns,0.5)

major_ticksY = np.arange(0, num_rows,1)
minor_ticksY = np.arange(0, num_columns,0.5)


#set minor and major axes ticks
ax.set_xticks(major_ticksX)
ax.set_xticks(minor_ticksX, minor=True)
ax.set_yticks(major_ticksY)
ax.set_yticks(minor_ticksY, minor=True)

# Axes labels
ax.set_xticklabels(sensorX)
ax.set_yticklabels(sensorY)

#invert order of y-axis, so that th eleft bottom corner is y=0
plt.gca().invert_yaxis()


 # Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
                  rotation_mode="anchor")
print(array_plot)
ax.grid(which='minor', alpha=0.2)
plt.colorbar(im)

ax.set_title("Lucas Pressure Sensor Array")
fig.tight_layout()'''
################Read  Linearizations coeeficients from .csv file#########
print('Reading linearization coefficients from .csv file')
with open('coefficients_S10N_14nodes_final_80Sensors.csv',newline='') as csvfile:
    coeff_data = list(csv.reader(csvfile, delimiter=','))
    coeff_ref[0]=coeff_data[1][0]
    coeff_ref[1]=coeff_data[1][1]
    xi= coeff_data[2][:]
    xd= coeff_data[3][:]
    yi= coeff_data[4:(4+num_sensors)][:]   
    yd= coeff_data[(4+num_sensors):(4+num_sensors+num_sensors)][:]
    print("len yd")
    print(len(yd))
    #convert string lists to float lists
    coeff_ref=[float(h) for h in coeff_ref]
    xi=[float(i) for i in xi]
    xd=[float(k) for k in xd]
    yi=[[float(j) for j in x] for x in yi]
    yd=[[float(l) for l in y] for y in yd]

    num_nodes=len(xi)
    print('Number of nodes')
    print(num_nodes)
    print('Reference Coefficients')
    print(coeff_ref)
    print('xi')
    print(xi)
    print('xd')
    print(xd)
    print('yi')
    print(yi[:][:])
    print('yd')
    print(yd[:][:])

################Action GUI Interface: ###################################
window = Tk()
window.title("Artificial Skin Array")
window.geometry('350x100+800+200')
lbl = Label(window, text="Press 'q' on keyboard to stop measurement!",font=("Arial Bold", 12))
lbl.pack(padx=10, pady=10, side=TOP)

############### Python program to get transpose#############################
# elements of two dimension list
def transpose(l1, l2):
 
    # iterate over list l1 to the length of an item
    for i in range(len(l1[0])):
        # print(i)
        row =[]
        for item in l1:
            # appending to new list with values and index positions
            # i contains index position and item contains values
            row.append(item[i])
        l2.append(row)
    return l2
####################  FUNCTIONS: save_data(), BUTTON FUNCTIONS: stop()and  record()#############
'''def save_data():

    print('Recording Paused - Saving Data in .CSV File')
    #calculating transposing so that data can be saved in same format as calibration data from Matlab
    transpose_data = []
    transpose_lin_data = []
    transpose(raw_data,  transpose_data)
    transpose(lin_data,  transpose_lin_data)
    #print(lin_data)
    #print(transpose_lin_data)

    #if(reading_count[1]==0):
    print('INTERMEDIATE SAVING - FIRST SAVE ACTION SINCE PROGRAM START')
    #writing data to .csv file:


    with open('pressure_raw.csv','w',newline='') as pressure_data: #open csv file in 'write' mode
        data_writer = csv.writer(pressure_data, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
        data_writer.writerow(['Exp'] + input_type) #type of tactile input
        data_writer.writerow(transpose_data[0][:])#time data
        data_writer.writerow([]) #empty row
        for b in range(1,num_sensors+1):
                data_writer.writerow(transpose_data[b][:]) #tactile data


    with open('pressure_lin.csv','w',newline='') as linearized_data: #open csv file in 'write' mode
        data_writer = csv.writer(linearized_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
        data_writer.writerow(['Exp'] + input_type) #type of tactile input
        data_writer.writerow(transpose_data[0][:]) #time data
        data_writer.writerow([]) #empty row
        for b in range(0,num_sensors):
            data_writer.writerow(transpose_lin_data[b][:]) #linearized tactile data
            '''



###################Stop Recording and Save %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def save_data():   

    #print(lin_data)
    lbl.configure(text="Process stopped!",font=("Arial Bold", 18))
    global stop_pressed
    stop_pressed = True
    print('Recording Stopped - Saving Data in .CSV File')

    #print('Transpose')
    #calculating transposing so that data can be saved in same format as calibration data from Matlab
    transpose_data = []
    transpose_lin_data = []
    transpose(raw_data,  transpose_data)
    transpose(lin_data,  transpose_lin_data)
    
    transpose_data = np.array(transpose_data)
    transpose_lin_data = np.array(transpose_lin_data)
    #print(transpose_lin_data.shape)

    #if(reading_count[1]==0):
    print('FINAL DATA SAVING - FIRST SAVE ACTION SINCE RECORDING START')
    #writing data to .csv file:
    #Save raw data to xlsx:
    file_raw=input_type + '_RAW.xlsx'
    file_raw='C:/Users/lucas.dahl/Documents/Artificial_Skin_Recording_and_Plot_Arm/Artificial_Skin_Recording_Arm/Pressure_Data/' + file_raw
    workbook = xlsxwriter.Workbook(file_raw)
    worksheet = workbook.add_worksheet()

    #print('Time_data')
    #print(transpose_data[0][:])
    worksheet.write(0,0, input_type)
    for col_num, data in enumerate(transpose_data[0][:]):
        worksheet.write(1, col_num, data)
    
    #print('Sensor 1 data raw')
    #print(transpose_data[1][:])
    for b in range(1,num_sensors+1):
        for col_num, data in enumerate(transpose_data[b][:]):
            worksheet.write(b+2, col_num, data)

    workbook.close()

    #Save linearized data to xlsx:
    file_lin=input_type + '_LIN.xlsx'
    file_lin='C:/Users/lucas.dahl/Documents/Artificial_Skin_Recording_and_Plot_Arm/Artificial_Skin_Recording_Arm/Pressure_Data/' + file_lin
    workbook = xlsxwriter.Workbook(file_lin)
    worksheet = workbook.add_worksheet()

    #print('Time_data')
    #print(transpose_data[0][:])
    #Time data
    worksheet.write(0,0, input_type)
    for col_num, data in enumerate(transpose_data[0][:]):
        worksheet.write(1, col_num, data)
    
    #print('Sensor 1 data lin')
    #print(transpose_lin_data[0][:])
    #Linearized data
    for b in range(0,num_sensors):
        for col_num, data in enumerate(transpose_lin_data[b][:]):
            worksheet.write(b+3, col_num, data)

    for b in range(num_sensors+1,num_sensors+3):
        for col_num, data in enumerate(transpose_data[b][:]):
            worksheet.write(b+3, col_num, data)
    workbook.close()
    sys.exit("Data saved - Program closed") 
    ''''
    #saves Exp. label in row 1, time in row 2, and sensor readings from row 4 inclusive
    with open('pressure_raw.csv','w',newline='') as pressure_data: #open csv file in 'write' mode
        raw_writer = csv.writer(pressure_data, delimiter=' ')        
        raw_writer.writerow(['Exp'] + input_type) #type of tactile input
        raw_writer.writerow(transpose_data[0][:])#time data
        raw_writer.writerow([]) #empty row
        #print(transpose_data[1][:])
        #transpose_data = np.array(transpose_data)
        #print(transpose_data[1][:])
        #raw_writer.writerow(transpose_data[1][:])
        for b in range(1,num_sensors+1):
                print(transpose_data[b][:])
                raw_writer.writerow(transpose_data[b][:]) #tactile data
                #raw_writer.writerow(['End Row']) #empty row

    
    # good idea to close if you're done with it
    pressure_data.close()

    with open('pressure_lin.csv','w',newline='') as linearized_data: #open csv file in 'write' mode
        data_writer = csv.writer(linearized_data, delimiter=' ')        
        data_writer.writerow(['Exp'] + input_type) #type of tactile input
        data_writer.writerow(transpose_data[0][:]) #time data
        data_writer.writerow([]) #empty row
        for b in range(0,num_sensors):
            print(transpose_lin_data[b][:])
            data_writer.writerow(transpose_lin_data[b][:]) #linearized tactile data

    
    linearized_data.close()  # good idea to close if you're done with it
    '''
    
    
    '''elif(reading_count[1]>0):
        print('FINAL DATA SAVING - NOT THE FIRST SAVE ACTION SINCE PROGRAM START')
        with open('pressure_raw.csv','a',newline='') as pressure_data: #open csv file in 'append' mode
            data_writer = csv.writer(pressure_data, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for b in range(1,num_sensors+1):
                data_writer.writerow(transpose_data[b][:]) #linearized tactile data

        pressure_data.close()  # good idea to close if you're done with it
        time.sleep(.1)
        with open('pressure_lin.csv','w',newline='') as linearized_data: #open csv file in 'write' mode
            data_writer = csv.writer(linearized_data, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)        
            for b in range(0,num_sensors):
                data_writer.writerow(transpose_lin_data[b][:]) #linearized tactile data
        
        linearized_data.close()  # good idea to close if you're done with it
        time.sleep(.1)
        sys.exit("Data saved - Program closed") '''
#######################Record data ################################################################################
def record(pressure_label, pressure_plot, pressure_recording):

    countr=0;
    countf=0;

    global stop_pressed
    stop_pressed = False   
    ser_input=[]
    #check if the array was calibrated this run. If not read the old calibration data from the csv. file and
    #send them to the arduino
    ser.flush()
    ser.write(b'R\n') # Tell Arduino that we DONT want to send the transfer functions but record directly
    recording_confirm=False
    while recording_confirm==False:
        ser_input = ser.readline()[:-2] #the last bit gets rid of the new-line chars       
        print(ser_input)
        ser_input = ser_input.decode('ascii')
        print(ser_input)
        if ser_input[0]=="R":
            print("Data recording confirmed")
            recording_confirm=True

    print("Data recording started")
    lbl.configure(text="Data recording started!",font=("Arial Bold", 18))
    ser.flush()
    ser_input=[]
    #previous_time = timer()
    #start_time = previous_time
    first = True
    
    while stop_pressed == False: #Makes a continuous loop to read values from Arduino    
        read=False
        if keyboard.is_pressed("q"):
            print("q pressed, ending recording")
            save_data()
            break
        while read==False:
            #print("previous_time")
            #print(previous_time)
            if(first==True):
                start_time = timer()
                previous_time = start_time
                first=False

            current_time=timer()
            #print("current_time")
            #print(current_time)
            
            #####READ SERIAL INPUT #######################################################
            if(current_time-previous_time>=time_step):
                #print(current_time-previous_time)
                time_interval=(current_time-start_time)
                previous_time=current_time
                time_interval = float("{:.2f}".format(time_interval)) #limit time_interval to two digits
                ser_input = ser.readline()[:-2] #the last bit gets rid of the new-line chars
                #print('Serial Input')
                #print(ser_input)
                read=True
                processing_time_start = time.time() 
    
        #######################Parse Serial INPUT ###############################################
        ser_input=ser_input.decode('ascii') #gets rif of the 'b' before the string
        ser_input = str(ser_input).split(",") #convert binary string to string and then split the string at each comma
        #ser_input=ser_input[0:-1] #remove last element from received serial input. It does not contain any pressure information
        current_pos=0
        ser_size =len(ser_input)
        
        #Check if we are recording more measurements than in 'num_measurements' specified. If so, save data and quit program
        if(reading_count[0]>=num_measurements):
            print('WARNING!!!MAX NUMBER OF SPECIFIED MEASUREMENTS (num_measurements) REACHED! SAVING DATA AND QUITING PROGRAM')
            save_data()

        #Assign current time interval to data
        raw_data[reading_count[0]][0] = time_interval
        lin_data[reading_count[0]][0] = time_interval
        for c in range(1,num_sensors+1):          
            raw_data[reading_count[0]][c] = ser_input[current_pos % ser_size]
            raw_data[reading_count[0]][c] = float(raw_data[reading_count[0]][c]) #convert data to float
            #print('------------------------------------')
            #print('Reading number:')
            #print(reading_count[0])
            #print('Sensor Number:')
            #print(c)
            #print('Sensor measurements:')
            #print(raw_data[reading_count[0]][c])
            if(raw_data[reading_count[0]][c]>511):
                print('Warning!!!!!! Entering overpressure range of sensor number 0x',I2C_address[c])
                winsound.Beep(4000, 2)
            if(raw_data[reading_count[0]][c]>760): #sensor range limit of S10N with a scaling factor of 2.4 is 767 corresponding to 15N. This is no the max limit the sensor can handle
                print('Warning!!!!!! Absolut sensor range limit of sensor number 0x',I2C_address[c])
                winsound.Beep(8000, 2)
            if(raw_data[reading_count[0]][c]==-250 or raw_data[reading_count[0]][c]==-251 ):
                print('Warning!!!!!! Faulty breakoutboard of sensor 0x',I2C_address[c-1])
            if(raw_data[reading_count[0]][c]==772):
                print('Warning!!!!!! Faulty connection of sensor 0x',I2C_address[c])
            current_pos+=1

        raw_data[reading_count[0]][num_sensors+1]=ser_input[current_pos % ser_size]
        raw_data[reading_count[0]][num_sensors+2]=ser_input[current_pos+1 % ser_size]
        #print('Arm position in  degrees:')
        #print(raw_data[reading_count[0]][num_sensors+2]) #print rotary encoder position of arm
        #print('Orthosis position in degrees:')
        #print(raw_data[reading_count[0]][num_sensors+3]) #print rotary encoder position of orthosis

        print('Reading Number:')
        print(reading_count[0])
        print('Raw data')
        print(raw_data[reading_count[0]])
        ##########################################################Pressure Linearization######################################################
        #if condition checking for falling or rising edge
        for sen in range(1,(num_sensors+1)):
            #print('-----------------------------New Measurement---------------------------------------')
            #print('Sensor number')
            #print(sen)
            segment=0
            #print('Reading Count')
            #print(reading_count[0]-1)
            #print(reading_count[0]-2)
            #print(reading_count[0]-3)

            if(reading_count[0]>=3):
                #print('Data and mean')               
                #print(raw_data[reading_count[0]-1][sen])
                #print(raw_data[reading_count[0]-2][sen])
                #print(raw_data[reading_count[0]-3][sen])
                data_mean = statistics.mean([raw_data[reading_count[0]-1][sen],raw_data[reading_count[0]-2][sen],raw_data[reading_count[0]-3][sen]])
                #print('Mean reading')
                #print(data_mean)
                #print('Current reading')
                #print(raw_data[reading_count[0]][sen])
        ##########################################################Check for Rising Signal######################################################
            #if prev. reading smaller than the current reading, then we have a rising signal
            if(reading_count[0]<3 or raw_data[reading_count[0]][sen]>=data_mean):
                #print('---Rising Pressure---')
                countr=countr+1;               
                for nd in range(0,num_nodes-1):
                    #print('node number')
                    #print(nd)
                    if(segment==0):
                        #find segement of the piecewise linear aprrox. in which the current sensor reading falls. Also considering the range below and above the range of the lowest and highest nodes
                        if((yi[sen-1][nd]<=raw_data[reading_count[0]][sen]<yi[sen-1][nd+1]) or (raw_data[reading_count[0]][sen]<yi[sen-1][0] and nd==0) or (raw_data[reading_count[0]][sen]>=yi[sen-1][num_nodes-1] and (nd+2)==num_nodes)):
                            #print('Reading range identified:')
                            #print(yi[sen-1][nd])
                            #print(yi[sen-1][nd+1])
                            #print('Coefficients')
                            #print(coeff_ref[0])
                            #print(coeff_ref[1])
                            #print(xi[nd])
                            #print(xi[nd+1])
                            #print(yi[sen-1][nd])
                            #print(yi[sen-1][nd+1])
                            segment=1
                            lin_data[reading_count[0]][sen-1]=(coeff_ref[0]*xi[nd]+coeff_ref[1]) + (((coeff_ref[0]*xi[nd+1]+coeff_ref[1]) - (coeff_ref[0]*xi[nd]+coeff_ref[1]))/(yi[sen-1][nd+1]-yi[sen-1][nd])) * (raw_data[reading_count[0]][sen]-yi[sen-1][nd])
                            #print('---linearized pressure:---')
                            #print(lin_data[reading_count[0]][sen-1])
            else:
                #print('---Falling Pressure---')
                countf=countf+1; 
                for nd in range(0,num_nodes-1):
                    #print('node number')
                    #print(nd)                                 
                    if(segment==0):
                        #find segement of the piecewise linear aprrox. in which the current sensor reading falls
                        if((yd[sen-1][nd]>=raw_data[reading_count[0]][sen]>yd[sen-1][nd+1]) or (raw_data[reading_count[0]][sen]>yd[sen-1][0] and nd==0) or (raw_data[reading_count[0]][sen]<=yd[sen-1][num_nodes-1] and (nd+2)==num_nodes)):
                            #print('Reading range identified:')
                            #print(yd[sen-1][nd])
                            #print(yd[sen-1][nd+1])
                            #print('Coefficients')
                            #print(coeff_ref[0])
                            #print(coeff_ref[1])
                            #print(xd[nd])
                            #print(xd[nd+1])
                            #print(yd[sen-1][nd])
                            #print(yd[sen-1][nd+1])
                            segment=1
                            lin_data[reading_count[0]][sen-1]=(coeff_ref[0]*xd[nd]+coeff_ref[1]) + (((coeff_ref[0]*xd[nd+1]+coeff_ref[1]) - (coeff_ref[0]*xd[nd]+coeff_ref[1]))/(yd[sen-1][nd+1]-yd[sen-1][nd])) * (raw_data[reading_count[0]][sen]-yd[sen-1][nd])
                            #print('---linearized pressure:---')
                            #print(lin_data[reading_count[0]][sen-1])

        reading_count[2]=reading_count[0] #store the number of current readings
        reading_count[0]+=1 #increment the number of current readings
        
        
        #print('countr')
        #print(countr)
        #print('countf')
        #print(countf)
        #print('------------------------------------')
        #After recording 1000 values save values to prevent data loss
        '''if(reading_count[0]%1000==0): #if num_reading is exactly divisible by 1000 (without remainder) then save the pressure data to the excel file
            print('INTERMEDIATE SAVING')
            print(reading_count[1])
            print(reading_count[0])            
            save_data()
            reading_count[1]=reading_count[0]'''

        #To prevent a program crash: Check if the number of received sensor values correspond to the in this code
        #specified number of sensors. If less sensor values are received fill the rest with zeros and print a waring
        num_readings = len(ser_input) #get number of list elements = number of sensors read
        #print('Number of Sensors Read:')
        #print(num_readings)
        if (num_readings<num_sensors+2):
            ser_input=(ser_input + [0]*num_sensors)[:num_sensors] #fill open array spotsto num_sensors with zeros
            print('WARNING!! LESS SENSOR READINGS RECEIVED THAN in "num_sensors" SPECIFIED')
            print(ser_input)
        elif (num_readings>num_sensors+2):
            print('WARNING!! MORE SENSOR READINGS RECEIVED THAN in "num_sensors" SPECIFIED')
        
        
        #######################################Plotting ##################################################
        #Plotting slows down the recording process and is therefore commented at the moment
        '''
        for i in range(0,num_sensors):
            #print(i)
            #print(ser_input[i])
            pressure_recording[i]=int((ser_input[i]))

        rows=pressure_plot.shape[0]
        columns=pressure_plot.shape[1]
        index=0
        for r in range(rows):
            for c in range(columns):
                pressure_plot[r][c]=pressure_recording[index]
                index = index+1

        #print("Decoded pressure data from Arduino")
        #print(pressure_plot)
        im.set_array(pressure_plot)
        
        #Delete previous annotations of the plot to prevent overwriting of the pressure values
        for ann in pressure_label:
            ann.remove()
        pressure_label[:]=[]

        #Loop over data dimensions and create text annotations.
        for i in range(len(sensorY)):
            for j in range(len(sensorX)):
                pressure_label.append(ax.text(j, i, pressure_plot[i, j],
                                    ha="center", va="center", color="w"))

        # allow some delay to render the image
        plt.pause(0.1)'''
    #plt.ioff()


############################## GUI BUTTONS ###############################################################
#btn_stop = Button(window, text="Stop Recording",font=("Arial Bold", 18), command=stop, bg="Blue", fg="black")
#btn_stop.pack(padx=10, pady=10, side=BOTTOM,fill=BOTH)

btn_record = Button(window, text="Record Data",font=("Arial Bold", 18), command= lambda: record(pressure_label, array_plot, array_pressure), bg="red", fg="black")
btn_record.pack(padx=10, pady=10, side=BOTTOM,fill=BOTH)


window.mainloop() #this function calls the endless loop of the window, so the window will wait for any user interaction till we close it.




