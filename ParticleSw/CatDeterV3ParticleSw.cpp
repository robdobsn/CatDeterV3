// Cat Deterrent V3
// Rob Dobson 2012-2017

/*
 * SYSTEM_MODE:
 *     - AUTOMATIC: Automatically try to connect to Wi-Fi and the Particle Cloud and handle the cloud messages.
 *     - SEMI_AUTOMATIC: Manually connect to Wi-Fi and the Particle Cloud, but automatically handle the cloud messages.
 *     - MANUAL: Manually connect to Wi-Fi and the Particle Cloud and handle the cloud messages.
 *
 * SYSTEM_MODE(AUTOMATIC) does not need to be called, because it is the default state.
 * However the user can invoke this method to make the mode explicit.
 * Learn more about system modes: https://docs.particle.io/reference/firmware/photon/#system-modes .
 */
#include "application.h"
#include "config.h"

#if defined(ARDUINO)
SYSTEM_MODE(SEMI_AUTOMATIC);
#endif

const int SPRAY_CTRL = D1;
const int LIGHTING_CTRL = D0;
const int DEFAULT_LIGHT_LEVEL = 0;
const int SPRAY_ON_TIME_LIMIT_MS = 2000;

// your network name also called SSID
char ssid[] = "rdint01";
// your network password
#if defined(WIFI_PASSWORD)
char password[] = WIFI_PASSWORD;
#else
char password[] = "";
#endif

// local port to listen on
unsigned int localPort = 7191;

//buffer to hold incoming packet
char packetBuffer[255];
char replyBuffer[] = "ACK0000000000\0\0\0\0";

UDP Udp;

void printWifiStatus() {
  // print the SSID of the network you're attached to:
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi IP address:
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength:
  long rssi = WiFi.RSSI();
  Serial.print("signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}

unsigned long lastSprayStartedMS = 0;

unsigned long responseIdx = 0;

static String __appStatusStr;

int loopLastMs = millis();

void spray(bool turnOn)
{
    digitalWrite(SPRAY_CTRL, turnOn ? 1 : 0);
}

void setup() {
  //Initialize serial and wait for port to open:
  Serial.begin(115200);

  // attempt to connect to Wifi network:
  Serial.print("Attempting to connect to Network named: ");
  // print the network name (SSID);
  Serial.println(ssid);

  // spray as output
  pinMode(SPRAY_CTRL, OUTPUT);
  spray(false);

  // Lighting as PWM OUTPUT
  pinMode(LIGHTING_CTRL, OUTPUT);
  analogWrite(LIGHTING_CTRL, DEFAULT_LIGHT_LEVEL);

  // Connect to WPA/WPA2 network. Change this line if using open or WEP network:
  WiFi.on();
  if (strlen(password) > 0)
      WiFi.setCredentials(ssid,password);
  WiFi.connect();

  while ( WiFi.connecting()) {
    // print dots while we wait to connect
    Serial.print(".");
    delay(300);
  }

  Serial.println("\nYou're connected to the network");
  Serial.println("Waiting for an ip address");

  while (WiFi.localIP() == INADDR_NONE) {
    // print dots while we wait for an ip addresss
    Serial.print(".");
    delay(300);
  }

  Serial.println("\nIP Address obtained");
  printWifiStatus();

  Serial.println("\nWaiting for a connection from a client...");
  Udp.begin(localPort);

  // Variable for application status
  Particle.variable("status", __appStatusStr);

}

void loop() {
    if ((millis() > loopLastMs + 30000) || (millis() < loopLastMs))
    {
        String ipStr = WiFi.localIP();
        Serial.printlnf("Cat Spray IP %s Port %d", ipStr.c_str(), localPort);
        loopLastMs = millis();

        byte macaddr[6];
        WiFi.macAddress(macaddr);
        __appStatusStr = "{'wifiIP':'" + ipStr + ",'wifiMAC':'";
        __appStatusStr += String::format("%02X:%02X:%02X:%02X:%02X:%02X", macaddr[0], macaddr[1], macaddr[2], macaddr[3], macaddr[4], macaddr[5]);
        __appStatusStr += "'}";

        if (Particle.connected())
            Particle.publish("CatDeter Alive", __appStatusStr);

    }

      if (Particle.connected() == false) {
    Particle.connect();
  }
  Particle.process();
  // if there's data available, read a packet
  int packetSize = Udp.parsePacket();
  if (packetSize) {
    Serial.print("Received packet size ");
    Serial.print(packetSize);
    Serial.print("bytes, from ");
    IPAddress remoteIp = Udp.remoteIP();
    Serial.print(remoteIp);
    Serial.print(", port ");
    Serial.println(Udp.remotePort());
    String ipStr = Udp.remoteIP();
    String msg = String::format("Rx %d from %s port %d", packetSize, ipStr.c_str(), Udp.remotePort());
    Particle.publish("DEBUG", msg);

    // read the packet into packetBufffer
    int len = Udp.read(packetBuffer, 255);
    if (len > 0) packetBuffer[len] = 0;
    Serial.printlnf("Contents:%s", packetBuffer);

    if (packetBuffer[0] == '0')
    {
        spray(false);
        Serial.println("Spray off - command");
    }
    else if (packetBuffer[0] == '1')
    {
        spray(true);
        lastSprayStartedMS = millis();
        Serial.println("Spray on - command");
    }
    else if (packetBuffer[0] == 'L')
    {
        // Ensure null terminated
        packetBuffer[10] = 0;
        uint8_t lightingLevel = atoi(packetBuffer+2);
        analogWrite(LIGHTING_CTRL, lightingLevel);
        Serial.printlnf("Lighting level %d", lightingLevel);
    }
    // send a reply, to the IP address and port that sent us the packet we received
    Udp.beginPacket(Udp.remoteIP(), Udp.remotePort());
    sprintf(replyBuffer+3, "%010d", responseIdx++);
    Udp.write(replyBuffer);
    Udp.endPacket();
  }

  // Check for spray needing to end
  if ((lastSprayStartedMS != 0) && (millis() > lastSprayStartedMS + SPRAY_ON_TIME_LIMIT_MS))
  {
      lastSprayStartedMS = 0;
      spray(false);
      Serial.println("Spray off - timeout");
  }
}
