#!/bin/bash
crontab -l > mycron
echo "@reboot sh /home/pi/rerun.sh" >> mycron
crontab mycron
rm mycron

