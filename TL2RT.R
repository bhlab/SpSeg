################################################################################
#Function and codes to convert a Timelapse output (.csv) to camtrapR recordTable,
#which can be then further used for various analyses.
#Author: Shivam Shrotriya
#Date: 03-11-2022
################################################################################

##Step1. Call R libraries
require(dplyr)
require(tidyr)

##Step2. Define the function TL2RT at the line no. 24 first. (Simply run this code without any changes) 
##Step3. Come back to line no. 17 to run the function with your inputs.

# Correct path to Timelapse.csv is required. Rest options can be changed as per requirement
#'relpath' requires the names of the folders (in same sequence) in RelativePath in timelapse output 
rt.output <- TL2RT("TimelapseData.csv", relpath = c("Station","Camera"), 
                   camerasIndependent = TRUE, minDeltaTime = 1, 
                   deltaTimeComparedTo = "lastIndependentRecord", 
                   timeZone = Sys.timezone(),
                   writecsv = TRUE, "outDir")

################ TL2RT Function###################
TL2RT <- function(TimelapseFile, relpath = c("Station","Camera"), 
                  camerasIndependent, minDeltaTime = 0, 
                  deltaTimeComparedTo, timeZone, 
                  writecsv = FALSE, outDir){
  wd0 <- getwd()
  on.exit(setwd(wd0))
  ### Assertions
  if (!is.character(TimelapseFile)) 
    stop("TimelapseFile must be of class 'character'", call. = FALSE)
  if (!file.exists(TimelapseFile)) 
    stop("Could not find TimelapseFile:\n", TimelapseFile, call. = FALSE)
  if (!hasArg(timeZone)) {
    message("timeZone is not specified. Assuming UTC")
    timeZone <- "UTC"
  }
  if (!is.element(timeZone, OlsonNames())) {
    stop("timeZone must be an element of OlsonNames()", call. = FALSE)
  }
  if (!is.vector(relpath))
    stop("RelativePath is not a vector", call. = FALSE)
  if (!"Station" %in% relpath)
    stop("Station is not defined in the RelativePath", call. = FALSE)
  if ("Camera" %in% relpath){
    if (!hasArg(camerasIndependent))
      stop("camerasIndependent is not defined. TRUE indicates that cameras on both flanks are independent", 
           call. = FALSE)
  } else {    
    message("Camera column not available. Setting camerasIndependent to FLASE")
    camerasIndependent = FALSE
  }
  if (!is.logical(camerasIndependent)) 
    stop("camerasIndependent must be of class 'logical'", call. = FALSE)
  if (hasArg(outDir)) {
    if (!is.character(outDir)) 
      stop("outDir must be of class 'character'", call. = FALSE)
    if (isFALSE(file.exists(outDir))) 
      stop("outDir does not exist", call. = FALSE)
  }
  minDeltaTime <- as.integer(minDeltaTime)
  if (!is.integer(minDeltaTime)) 
    stop("'minDeltaTime' must be an integer", call. = FALSE)
  if (minDeltaTime != 0) {
    deltaTimeComparedTo < match.arg(deltaTimeComparedTo, 
                                    choices = c("lastRecord", "lastIndependentRecord"))
    if (!hasArg(deltaTimeComparedTo)) {
      stop(paste("minDeltaTime is not 0. deltaTimeComparedTo must be defined"), 
           call. = FALSE)
    }
    message("minDeltaTIme is not 0. Duplicate records will be removed")
  } else {
    if (hasArg(deltaTimeComparedTo)) {
      warning(paste("minDeltaTime is 0. deltaTimeComparedTo = '", 
                    deltaTimeComparedTo, "' will have no effect", 
                    sep = ""), call. = FALSE, immediate. = TRUE)
    } else {
      deltaTimeComparedTo <- "lastRecord"
    }
  }
  if (!is.logical(writecsv)) 
    stop("writecsv must be logical (TRUE or FALSE)", call. = FALSE)
  
  ### Internal functions
  dt.logic <- function(x){
    y <- TRUE
    if (length(x)>1){
      for (i in 1:(length(x)-1)){
        dt <- difftime(x[i+1], x[max(which (y==TRUE))],tz = timeZone, units = "mins")
        y <- c(y, (minDeltaTime < as.numeric(dt)))
      }
    }
    return(y)
  }
  delta.time <- function(x,units){
    y <- 0
    if (length(x)>1){
      for (i in (1:length(x)-1)){
        dt <- difftime(x[i+1],x[i],tz = timeZone, units)
        y <- c(y,as.numeric(dt))
      }
    }
    return(format(round(as.numeric(y),2)))
  }
  
  ### Data preparation
  tl.dat <- read.csv(TimelapseFile)
  if (nrow(tl.dat) <= 1) 
    stop("TimelapseFile may only consist of 1 element only", call. = FALSE)
  tl.dat <- tl.dat %>% separate(RelativePath, relpath, remove = FALSE)
  tags <- c("Person", "Animal", "Empty", "Vehicle")
  for (i in 1:4){
    if (!is.logical (tl.dat[,tags[i]])){
      tl.dat[,tags[i]] <- tl.dat[,tags[i]] == "true" | tl.dat[,tags[i]] == "TRUE"
    }
  }
  wcount <- 0
  for (i in 1:nrow(tl.dat)){
    if(nchar(tl.dat$Species[i]) == 0 && tl.dat$Animal[i] == TRUE){
      warning(paste(tl.dat$File[i],"in",tl.dat$RelativePath[i],"is missing species identification."), 
              call. = FALSE)
      wcount <- wcount+1
    }
    if(tl.dat$Person[i] == FALSE && tl.dat$Animal[i] == FALSE && tl.dat$Empty[i] == FALSE && tl.dat$Vehicle[i] == FALSE){
      warning(paste(tl.dat$File[i],"in",tl.dat$RelativePath[i],"has all the tags set to false."), 
              call. = FALSE)
      wcount <- wcount+1
    }
    
  }
  if (wcount >0)
    stop(paste("There are", wcount, "errors in the species tagging."), call. = FALSE)
  Species <- NULL
  for (i in 1:nrow(tl.dat)){
    Species[i] <- ifelse (nchar(tl.dat$Species[i]) >0, tl.dat$Species[i],
                    ifelse(tl.dat$Vehicle[i] == TRUE, "vehicle",
                           ifelse(tl.dat$Person[i] == TRUE, "person", "blank")))
  }
  tl.dat$Species <- Species
  tl.dat$DateTimeOriginal <- as.POSIXct(strptime(x = tl.dat$DateTime, 
                                                 format = "%Y-%m-%d %H:%M:%S", tz = timeZone))
  
  ### Main
  if (minDeltaTime != 0){
    if (deltaTimeComparedTo == "lastIndependentRecord") {
      if (camerasIndependent) {
        record.table <- tl.dat %>% group_by(Station, Species, Camera) %>% 
          arrange(Station, Camera, Species, DateTimeOriginal) %>% filter (
            dt.logic(DateTimeOriginal))
      } else {
        record.table <- tl.dat %>% group_by(Station, Species) %>% 
          arrange(Station, Species, DateTimeOriginal) %>% filter (
            dt.logic(DateTimeOriginal))
      }
    } else {
      if (camerasIndependent) {
        record.table <- tl.dat %>% group_by(Station, Camera) %>% 
          arrange(Station, Camera, Species, DateTimeOriginal) %>% filter (
            dt.logic(DateTimeOriginal))
      } else {
        record.table <- tl.dat %>% group_by(Station) %>% 
          arrange(Station, Species, DateTimeOriginal) %>% filter (
            dt.logic(DateTimeOriginal))
      }
    }
  } else {
    record.table <- tl.dat %>% group_by(Station) %>% 
      arrange(Station, Species, DateTimeOriginal)
  }
  record.table <- record.table %>% mutate(
    Date = as.Date(DateTimeOriginal, format = "%Y/%M/%d", tz = timeZone),
    Time = strftime(DateTimeOriginal, format = "%H:%M:%S",tz = timeZone),
    delta.time.secs = delta.time(DateTimeOriginal,units = "secs"),
    delta.time.mins = delta.time(DateTimeOriginal,units = "mins"),
    delta.time.hours = delta.time(DateTimeOriginal,units = "hours"),
    delta.time.days = delta.time(DateTimeOriginal,units = "days")) %>% ungroup
  
  if ("Camera" %in% relpath){
    record.table2 <- record.table %>% select(Station, Camera, Species, DateTimeOriginal, Date, Time, 
                                             delta.time.secs, delta.time.mins, delta.time.hours, 
                                             delta.time.days, Folder, RelativePath, File)
  } else {
    record.table2 <- record.table %>% select(Station, Species, DateTimeOriginal, Date, Time, 
                                             delta.time.secs, delta.time.mins, delta.time.hours, 
                                             delta.time.days, Folder, RelativePath, File)
  }
  record.table2 <- data.frame(record.table2, stringsAsFactors = FALSE, check.names = TRUE)
  c.tl <- count(tl.dat, Station)
  c.rt <- count(record.table2, Station)
  dups <- c.tl$n - c.rt$n
  for (i in 1:length(dups)){
    message(paste("Station",c.tl$Station[i], ": removed", dups[i], "duplicated images."))
  }
  if (writecsv == TRUE) {
    outtable_filename <- paste("TL_record_table_", minDeltaTime, 
                               "min_deltaT_", Sys.Date(), ".csv", sep = "")
    if (hasArg(outDir))
      setwd(outDir)
    message("saving csv to  ", file.path(getwd(), outtable_filename))
    write.csv(record.table2, file = outtable_filename)
  }
  return(record.table2)
}
############## END ##############