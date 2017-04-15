/*! \file Project.h
	\date Wednesday August 8, 2009
	\author Gregory Diamos <gregory.diamos@gatech.edu>
	\brief The header file for the Project family of cuda functions.
*/

#ifndef PROJECT_H_INCLUDED
#define PROJECT_H_INCLUDED

#include <stdio.h>

#include <redfox/ra/interface/Tuple.h>

//#include <hydrazine/cuda/Cuda.h>
//#include <hydrazine/interface/macros.h>
//#include <hydrazine/interface/debug.h>

#ifdef REPORT_BASE
#undef REPORT_BASE
#endif

#define REPORT_BASE 0

namespace ra
{

namespace cuda
{

__device__ void wrapdate(int& year, int& month)
{
	if(month > 12)
	{
		int inc_year = (month - 1) / 12;
		year += inc_year;
		month = month % 12;
	}
}

__device__ void int2date(unsigned long long int seconds, int& year, int& month, int& day)
{
	unsigned long long int days = seconds / 86400 - 8035;
	unsigned int year_int = 1992;
	unsigned int month_int = 0;

// 92
	if(days >= 366)
	{
		days -= 366;
		year_int ++;
	}
	else
		goto MONTH;
//93
	if(days >= 365)
	{
		days -= 365;
		year_int ++;
	}
	else
		goto MONTH;

//94
	if(days >= 365)
	{
		days -= 365;
		year_int ++;
	}
	else
		goto MONTH;
//95
	if(days >= 365)
	{
		days -= 365;
		year_int ++;
	}
	else
		goto MONTH;

//96
	if(days >= 366)
	{
		days -= 366;
		year_int ++;
	}
	else
		goto MONTH;
//97
	if(days >= 365)
	{
		days -= 365;
		year_int ++;
	}
	else 
		goto MONTH;
MONTH:
	int month_days[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
	
	if(year_int == 1992 || year_int == 1996)
		for(int i = 2; i < 13; ++i)
			month_days[i]++;

	for(int i = 1; i < 13; ++i)
	{
		if(days < month_days[i] && days >= month_days[i - 1])
		{
			month_int = i;
			days -= month_days[i - 1];
			break;	
		}
	}

	year = year_int;
	month = month_int;
	day = days + 1;

	return; 
}

__device__ unsigned long long int date2int(int year_int, int month_int, int date_int)
{
	unsigned long long int days = (year_int - 1970) * 365;

	if(month_int > 1)
	    days += 31;
	if(month_int > 2)
	    days += 28;
	if(month_int > 3)
	    days += 31;
	if(month_int > 4)
	    days += 30;
	if(month_int > 5)
	    days += 31;
	if(month_int > 6)
	    days += 30;
	if(month_int > 7)
	    days += 31;
	if(month_int > 8)
	    days += 31;
	if(month_int > 9)
	    days += 30;
	if(month_int > 10)
	    days += 31;
	if(month_int > 11)
	    days += 30;
	
	days += (date_int - 1);

        if(year_int == 1992 && month_int <= 2)
            days += 5;
        if(year_int == 1992 && month_int > 2)
            days += 6;
        if(year_int > 1992 && year_int < 1996)
            days += 6;
        if(year_int == 1996 && month_int <= 2)
            days += 6;
        if(year_int == 1996 && month_int > 2)
            days += 7;
        if(year_int > 1996)
            days += 7;

//printf("days %llu\n", days);

	return days * 86400;
}
}

}

#endif

