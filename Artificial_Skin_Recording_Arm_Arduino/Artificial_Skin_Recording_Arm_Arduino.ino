
/******************************************************************************
Assessment Device - Artificial Skin Recording (with SingleTact Sensors)

//Author: Lucas Dahl
//Date:08/02/2022 (last change)
/*
      __
   __/ o \           - -  -
  (__     \______
     |    __)   /  - -  - - -
      \________/
         _/_ _\_    - - -

*/
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Wire.h> //For I2C/SMBus
#include <SPI.h> //For SPI bus

//SPI BUS varibales
/* Serial rates for UART */
//#define BAUDRATE        115200
/* SPI commands */
#define AMT22_NOP       0x00
#define AMT22_RESET     0x60
#define AMT22_ZERO      0x70
/* Define special ascii characters */
#define NEWLINE         0x0A
#define TAB             0x09
/* We will use these define macros so we can write code once compatible with 12 or 14 bit encoders */
#define RES12           12
#define RES14           14
/* SPI pins */
#define ENC_Arm          5 //assessment device pin/cs
#define ENC_Orth         6 //orthosis pin/cs
#define SPI_MOSI        51
#define SPI_MISO        50 
#define SPI_SCLK        52

//ENCODER Variables
uint16_t encoderPosition; //create a 16 bit variable to hold the encoders position
uint16_t currentPosition[2]={0,0};
int minRange = 0; //minimum angle of full movement range in degrees
int maxRange = 90; //maximum angle of full movement range in degrees
uint16_t minPosition[2] = {3526, 11614}; //ecoder values corresponding to the minimum angles of arm and orthosis (from calibration with script " CUI_Encoder_ATM22_SPI_14bit_Calibration")
uint16_t maxPosition[2] = {180, 14949};  //ecoder values corresponding to the maximum angles of arm and orthosis (from calibration with script " CUI_Encoder_ATM22_SPI_14bit_Calibration")

//General variables - Please Enter the correct number and the desired readin speed in ms
const unsigned int NUM_SENSORS = 80;//48
unsigned long previousMillis;
unsigned long currentMillis;
unsigned long timeStep = 500; // period used to record and send the pressure data to the pc in ms

//I2C and SingleTact pressure sensors variables
byte i2cAddress[]{0x05,0x06,0x07,0x08,0x09,0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,0x19,0x20,0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28,0x29,0x30,0x31,0x32,0x33,0x34,0x35,0x36,0x37,0x38,0x39,0x40,0x41,0x42,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x50,0x51,0x52,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x60,0x61,0x62,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x70,0x71,0x72,0x73,0x74,0x75,0x0F,0x77,0x78,0x79,0x0A,0x0B,0x0C,0x0D,0x0E};
//byte i2cAddress[]{0x05,0x63,0x70,0x71,0x72,0x73,0x74,0x75,0x76,0x77,0x78,0x79};//,0x0A,0x0B,0x0C,0x0D,0x0E};

//Back bottom test array:{0x76, 0x78, 0x75, 0x77}; // Slave address (SingleTact), default 0x04
float data[NUM_SENSORS];
float offset[NUM_SENSORS];//{0,0,0}; //sensor baseline offsets which need to be substracted from all measurements

//Serial Communication variables
float incoming[NUM_SENSORS];
byte i = 0;
char record[100];
char recvchar;
byte indx = 0;
boolean new_data = false;
String received;
// strings for reading input
String commandString, valueString, indexString;
char char1[8];


void setup() {
  //I2C initialization
  Wire.begin(); // join i2c bus (address optional for master)
  //Wire.setClock(400000);
  //TWBR = 12; //Increase i2c speed if you have Arduino MEGA2560, not suitable for Arduino UNO
  Serial.begin(57600);  // start serial for output 57600

  //Init SPI interface: Set the modes for the SPI IO
  pinMode(SPI_SCLK, OUTPUT);
  pinMode(SPI_MOSI, OUTPUT);
  pinMode(SPI_MISO, INPUT);
  pinMode(ENC_Arm, OUTPUT);
  pinMode(ENC_Orth, OUTPUT); 
  //Get the CS line high which is the default inactive state
  digitalWrite(ENC_Arm, HIGH);
  digitalWrite(ENC_Orth, HIGH); 
  //set the clockrate. Uno clock rate is 16Mhz, divider of 32 gives 500 kHz.
  //500 kHz is a good speed for our test environment
  //SPI.setClockDivider(SPI_CLOCK_DIV2);   // 8 MHz
  //SPI.setClockDivider(SPI_CLOCK_DIV4);   // 4 MHz
  //SPI.setClockDivider(SPI_CLOCK_DIV8);   // 2 MHz
  //SPI.setClockDivider(SPI_CLOCK_DIV16);  // 1 MHz
  SPI.setClockDivider(SPI_CLOCK_DIV32);    // 500 kHz
  //SPI.setClockDivider(SPI_CLOCK_DIV64);  // 250 kHz
  //SPI.setClockDivider(SPI_CLOCK_DIV128); // 125 kHz 
  //start SPI bus
  SPI.begin();
  
  //Sensor offset elimination
  //offsetElimination();
  readInput();//pauses programm until some message from the PC is received
  if(received == "R"){
    //Serial.print("Arduino in Rec Mode");
    Serial.println(received);
    record_data();
  }
  else{
    Serial.println("No valid command (R)");
    Serial.println(received);
  }
}
void loop() {
  
  Serial.flush();
  readInput();//pauses programm until some message from the PC is received
 if(received == "R"){
    Serial.println(received);
    record_data();
  }
  else{
    Serial.println("No valid command (R)");
  }
}
//////////////////////////////////////////////////////////////////////////////////////////////////////
void record_data(){   
    while(true){
        currentMillis = millis();
        if (currentMillis - previousMillis >= timeStep) 
        {
          previousMillis = currentMillis;
          
          for(int i=0; i<NUM_SENSORS; i++){
            //Serial.println(i2cAddress[i],HEX);
            data[i]= (readDataFromSensor(i2cAddress[i]));
            data[i]=(data[i]-250);//*2.4;
            //data[i]=data[i]-offset[i];
            Serial.print(data[i]);
            Serial.print(',');
          }
        currentPosition[0] = getPositionSPI(ENC_Arm, RES14); //read assessment device (Elbow) angle
        currentPosition[1] = getPositionSPI(ENC_Orth, RES14); //read orthosis angle
        //map digital encoder values to angles in degrees:
        currentPosition[0]=map(currentPosition[0],minPosition[0],maxPosition[0],minRange,maxRange);
        currentPosition[1]=map(currentPosition[1],minPosition[1],maxPosition[1],minRange,maxRange);
        
        Serial.print(currentPosition[0]);
        Serial.print(',');
        Serial.println(currentPosition[1]);
        //Serial.println(']');
      }
    }
  memset(data, 0, sizeof(data)); //reset "data_sum" before doinf next measurement series
}
short readDataFromSensor(short address)
{
  byte i2cPacketLength = 6;//i2c packet length. Just need 6 bytes from each slave
  byte outgoingI2CBuffer[3];//outgoing array buffer
  byte incomingI2CBuffer[6];//incoming array buffer

  outgoingI2CBuffer[0] = 0x01;//I2c read command
  outgoingI2CBuffer[1] = 128;//Slave data offset
  outgoingI2CBuffer[2] = i2cPacketLength;//require 6 bytes

  Wire.beginTransmission(address); // transmit to device 
  Wire.write(outgoingI2CBuffer, 3);// send out command
  byte error = Wire.endTransmission(); // stop transmitting and check slave status
  if (error != 0) return -1; //if slave not exists or has error, return -1
  Wire.requestFrom(address, i2cPacketLength);//require 6 bytes from slave
  byte incomeCount = 0;
  while (incomeCount < i2cPacketLength)    // slave may send less than requested
  {
    if (Wire.available())
    {
      incomingI2CBuffer[incomeCount] = Wire.read(); // receive a byte as character
      incomeCount++;
    }
    else
    {
      delayMicroseconds(10); //Wait 10us 
    }
  }

  short rawData = (incomingI2CBuffer[4] << 8) + incomingI2CBuffer[5]; //get the raw data
  return rawData; 
}

void offsetElimination(){
  
  //1. offset elimination
  //Offset elimination: Take sensor output without load and store it as offset value
  for(int i=0; i<NUM_SENSORS; i++){
    //Take ten measurements of each sensor and calculate the average
    for(int j=0; j<10; j++){
      data[i] += readDataFromSensor(i2cAddress[i]);
      delay(100);
    }
  data[i]= data[i]/10; //calculate average of the last 10 sensor measurements
  offset[i]=data[i];
  }
}
////read Input via serial monitor//////////////
void readInput(){
   new_data = false;
   //Serial.println("Waiting for User Input");
   while(new_data == false){
       if (Serial.available())
    {
        recvchar = Serial.read();
        if (recvchar != '\n')
        { 
            record[indx++] = recvchar;
        }
        else if (recvchar == '\n')
        {
          record[indx] = '\0';
          indx = 0;
          new_data = true;
          i=0;

          //set the mode:
          //'R' = Record raw tactile data and send raw data to PC
          
          if (record[0]=='R'){
            //Serial.print("Record123");
            received=record[0]; //mode for main loop
          }
          
        }
      }
   }
}

////sub funct. of readInput to interpret read data package//////////
void getData(char record[])
{
    i = 0;
   
    char *index = strtok(record, ",");
    while(index != NULL)
    {
        incoming[i++] = atof(index); 
        index = strtok(NULL, ",");
    }
}

//SPI Functions
uint16_t getPositionSPI(uint16_t encoder, uint8_t resolution)
{
  uint16_t currentPosition;       //16-bit response from encoder
  bool binaryArray[16];           //after receiving the position we will populate this array and use it for calculating the checksum

  //get first byte which is the high byte, shift it 8 bits. don't release line for the first byte
  currentPosition = spiWriteRead(AMT22_NOP, encoder, false) << 8;   

  //this is the time required between bytes as specified in the datasheet.
  //We will implement that time delay here, however the arduino is not the fastest device so the delay
  //is likely inherantly there already
  delayMicroseconds(3);

  //OR the low byte with the currentPosition variable. release line after second byte
  currentPosition |= spiWriteRead(AMT22_NOP, encoder, true);        

  //run through the 16 bits of position and put each bit into a slot in the array so we can do the checksum calculation
  for(int i = 0; i < 16; i++) binaryArray[i] = (0x01) & (currentPosition >> (i));

  //using the equation on the datasheet we can calculate the checksums and then make sure they match what the encoder sent
  if ((binaryArray[15] == !(binaryArray[13] ^ binaryArray[11] ^ binaryArray[9] ^ binaryArray[7] ^ binaryArray[5] ^ binaryArray[3] ^ binaryArray[1]))
          && (binaryArray[14] == !(binaryArray[12] ^ binaryArray[10] ^ binaryArray[8] ^ binaryArray[6] ^ binaryArray[4] ^ binaryArray[2] ^ binaryArray[0])))
    {
      //we got back a good position, so just mask away the checkbits
      currentPosition &= 0x3FFF;
    }
  else
  {
    currentPosition = 0xFFFF; //bad position
  }

  //If the resolution is 12-bits, and wasn't 0xFFFF, then shift position, otherwise do nothing
  if ((resolution == RES12) && (currentPosition != 0xFFFF)) currentPosition = currentPosition >> 2;

  return currentPosition;
}

/*
 * This function does the SPI transfer. sendByte is the byte to transmit. 
 * Use releaseLine to let the spiWriteRead function know if it should release
 * the chip select line after transfer.  
 * This function takes the pin number of the desired device as an input
 * The received data is returned.
 */
uint8_t spiWriteRead(uint8_t sendByte, uint8_t encoder, uint8_t releaseLine)
{
  //holder for the received over SPI
  uint8_t data;

  //set cs low, cs may already be low but there's no issue calling it again except for extra time
  setCSLine(encoder ,LOW);

  //There is a minimum time requirement after CS goes low before data can be clocked out of the encoder.
  //We will implement that time delay here, however the arduino is not the fastest device so the delay
  //is likely inherantly there already
  delayMicroseconds(3);

  //send the command  
  data = SPI.transfer(sendByte);
  delayMicroseconds(3); //There is also a minimum time after clocking that CS should remain asserted before we release it
  setCSLine(encoder, releaseLine); //if releaseLine is high set it high else it stays low
  
  return data;
}

/*
 * This function sets the state of the SPI line. It isn't necessary but makes the code more readable than having digitalWrite everywhere 
 * This function takes the pin number of the desired device as an input
 */
void setCSLine (uint8_t encoder, uint8_t csLine)
{
  digitalWrite(encoder, csLine);
}

/*
 * The AMT22 bus allows for extended commands. The first byte is 0x00 like a normal position transfer, but the 
 * second byte is the command.  
 * This function takes the pin number of the desired device as an input
 */
void setZeroSPI(uint8_t encoder)
{
  spiWriteRead(AMT22_NOP, encoder, false);

  //this is the time required between bytes as specified in the datasheet.
  //We will implement that time delay here, however the arduino is not the fastest device so the delay
  //is likely inherantly there already
  delayMicroseconds(3); 
  
  spiWriteRead(AMT22_ZERO, encoder, true);
  delay(250); //250 second delay to allow the encoder to reset
}

/*
 * The AMT22 bus allows for extended commands. The first byte is 0x00 like a normal position transfer, but the 
 * second byte is the command.  
 * This function takes the pin number of the desired device as an input
 */
void resetAMT22(uint8_t encoder)
{
  spiWriteRead(AMT22_NOP, encoder, false);

  //this is the time required between bytes as specified in the datasheet.
  //We will implement that time delay here, however the arduino is not the fastest device so the delay
  //is likely inherantly there already
  delayMicroseconds(3); 
  
  spiWriteRead(AMT22_RESET, encoder, true);
  
  delay(250); //250 second delay to allow the encoder to start back up
}
