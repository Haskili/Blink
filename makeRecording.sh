# Change to the capture directory
cd "captures"

# Get each unique device name and output to log file
ls | egrep "DEV\_[0-9]+\-CAP_[0-9]+\.png" | awk -F '-' {'print $1'} \
   | uniq -c | awk -F ' ' {'print $2'} > devices.log

# For each group of files relating to a device
while IFS= read -r device
do

	# Check for existing video files of that device
	echo "Writing video for '$device'..."
	if [[ $(ls | grep "$device-recording.mp4") != "" ]]
		then
			echo "WARNING: Overwriting video file '$device-recording.mp4'!"
			rm -f "$device-recording.mp4"
	fi

	# Use ffmpeg to write video with capture files relating 
	# to the current '$device'
	ffmpeg -r 10 -f image2 -start_number 1 -i "$device-CAP_%d.png" \
		-c:v libx264 -vf fps=25 -pix_fmt yuv420p "$device-recording.mp4" \
		&>> makeRecording.log < /dev/null

	echo -e "Finished writing for '$device', moving on...\n"

done < "devices.log"
echo "Completed! Please check 'makeRecording.log' for information and output."
rm "devices.log"
cd ..