#!/bin/bash

# ScaleXI AI Abstract Dataset - Repository Update Script
# This script automates the process of updating the GitHub repository
# It requires manual input for commit message and version

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== ScaleXI AI Abstract Dataset Repository Update ===${NC}"
echo -e "${YELLOW}This script will update the GitHub repository with your changes.${NC}"
echo

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}Git is not installed. Please install Git first.${NC}"
    exit 1
fi

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree &> /dev/null; then
    echo -e "${YELLOW}Not in a Git repository. Please run this script from the repository root.${NC}"
    exit 1
fi

# Check if remote repository exists
if ! git remote -v | grep -q "github.com/scalexi/scalexi-ai-abstract-dataset"; then
    echo -e "${YELLOW}Remote repository not properly configured.${NC}"
    echo "Setting up remote repository..."
    git remote add origin https://github.com/scalexi/scalexi-ai-abstract-dataset.git
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Failed to add remote repository. Please check your configuration.${NC}"
        exit 1
    fi
fi

# Prompt for commit message
echo -e "${BLUE}Enter commit message:${NC}"
read -p "> " COMMIT_MESSAGE
if [ -z "$COMMIT_MESSAGE" ]; then
    echo -e "${YELLOW}Commit message cannot be empty. Using default message.${NC}"
    COMMIT_MESSAGE="Update ScaleXI AI Abstract Dataset"
fi

# Prompt for version
echo -e "${BLUE}Enter version (e.g., v1.0.0):${NC}"
read -p "> " VERSION
if [ -z "$VERSION" ]; then
    echo -e "${YELLOW}Version cannot be empty. Using date-based version.${NC}"
    VERSION="v$(date +%Y.%m.%d)"
fi

# Display files to be committed
echo -e "\n${BLUE}Files to be updated:${NC}"
git status --porcelain

# Confirm before proceeding
echo
echo -e "${YELLOW}Ready to update with the following settings:${NC}"
echo -e "  Commit message: ${COMMIT_MESSAGE}"
echo -e "  Version tag: ${VERSION}"
echo
read -p "Proceed with update? (y/n) " CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Update cancelled.${NC}"
    exit 0
fi

# Add all changes
echo -e "\n${BLUE}Adding all changes...${NC}"
git add .
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to add changes. Check for errors above.${NC}"
    exit 1
fi

# Commit changes
echo -e "\n${BLUE}Committing changes...${NC}"
git commit -m "$COMMIT_MESSAGE"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to commit changes. Check for errors above.${NC}"
    exit 1
fi

# Add tag with version
echo -e "\n${BLUE}Adding version tag...${NC}"
git tag -a "$VERSION" -m "Version $VERSION"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to add tag. Check for errors above.${NC}"
    # Continue anyway, not critical
fi

# Push changes
echo -e "\n${BLUE}Pushing changes to GitHub...${NC}"
git push origin main
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to push changes. You might need to pull first or set upstream.${NC}"
    echo -e "Trying to set upstream and push..."
    git push --set-upstream origin main
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}Still failed to push. Check your GitHub permissions and authentication.${NC}"
        exit 1
    fi
fi

# Push tags
echo -e "\n${BLUE}Pushing version tag...${NC}"
git push origin "$VERSION"
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Failed to push tag. Check your GitHub permissions.${NC}"
    # Continue anyway, not critical
fi

echo -e "\n${GREEN}Repository successfully updated!${NC}"
echo -e "Commit: ${COMMIT_MESSAGE}"
echo -e "Version: ${VERSION}"
echo -e "\n${BLUE}Visit your repository:${NC} https://github.com/scalexi/scalexi-ai-abstract-dataset"
echo 