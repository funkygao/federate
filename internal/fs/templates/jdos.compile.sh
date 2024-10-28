set -x

# Function to display error and exit
error_exit() {
    echo "Error: \$1" >&2
    exit 1
}

cp bin/federate /usr/bin
federate microservice scaffold
ENV=test
for repo in $(federate components); do
    git config -f .gitmodules submodule.$repo.branch $ENV
done

GIT_SSH_COMMAND='ssh -o StrictHostKeyChecking=no' git submodule update --init --recursive

federate microservice fusion-start || error_exit "Failed to scaffold {{.Name}}-starter"
federate microservice consolidate --yes --silent=true || error_exit "Failed to consolidate the target project"

# mvn install components and starter jar
for repo in $(federate components); do
    profile=$(federate inventory -f maven-profile -r $repo -e $ENV)
    modules=$(federate inventory -f maven-modules -r $repo)
    (cd $repo && mvn install -q -pl ":$modules" -P"$profile" -am -T8 -Dmaven.test.skip=true -Dfederate.packaging=true) || error_exit "Failed to install $repo"
done
(cd {{.Name}}-starter && mvn install -q -Dmaven.test.skip=true) || error_exit "Failed to install {{.Name}}-starter.jar"

# 最后一行承受 -f $(pwd) -T 1C -Dmaven.artifact.threads=16
mvn validate -q
