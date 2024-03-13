# inside soy directory, do...
# find . -type f -exec sh /home/hoppip/llvm-project-pistachio/EMILY-NOTES/replaceToy.sh "{}" \;

echo $1
sed -i 's+TOY+SOY+g' "$1"
sed -i 's+Toy+Soy+g' "$1"
sed -i 's+toy+soy+g' "$1"
exit