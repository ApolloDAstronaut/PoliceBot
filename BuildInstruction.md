# How to build the project:
1. ## Local Server
    1. You'll want to save your changes to git
        1. ```git add .``` to stage all files
        2. ```git commit -m "[INFORMATIVE MESSAGE]"``` to explain what you are sending
        3. ```git push``` to finally send everything
    2. Connect to remote server via ssh
       ```bash
        ssh user@rpi-server.local
       ```
       with the password ```quackquack```
2. ## Remote Server
    1. You'll want to get your changes from git
        1. ```git pull``` to get latest changes on main branch
    2. Run ```make build```
    3. Run ```make run```
3. ## Docker Container Within Remote Server
    1. Launch the necessary ros2 packages using
       ```bash
       ros2 run package node
       ```
       ```bash
       ros2 launch launchPackage launchFile
       ```
   
