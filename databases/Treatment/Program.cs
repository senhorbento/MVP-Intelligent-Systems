using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

public class Program
{
    public class WeatherData
    {
        public DateTime Data { get; set; }
        public TimeSpan Hora { get; set; }
        public float Chuva { get; set; }
        public float Nivel { get; set; }
        public float Temperatura { get; set; }
        public float Umidade { get; set; }
    }
    public static void Main()
    {
        const string originFilePath = "../2023-Nivel-Temperatura.csv";
        const string outputFilePath = "../2023-ByDay.csv";
        List<WeatherData> weatherDataList = ReadWeatherData(originFilePath);

        var dailyMetrics = weatherDataList
            .GroupBy(w => w.Data.Date)
            .Select(g => new
            {
                Date = g.Key,
                Avg_Temperature = CalculateAverageExcludingMinMax(g.Select(w => w.Temperatura)),
                Max_Humidity = g.Max(w => w.Umidade),
                Max_Preciptation = g.Max(w => w.Chuva),
                Level = g.First().Nivel
            })
            .ToList();

        using StreamWriter writer = new(outputFilePath);
        writer.WriteLine("Data;Avg_Temperature;Max_Humidity;Max_Preciptation;Level");
        foreach (var metrics in dailyMetrics)
        {
            writer.WriteLine($"{metrics.Date.ToShortDateString()};" +
                             $"{metrics.Avg_Temperature:F2};" +
                             $"{metrics.Max_Humidity:F2};" +
                             $"{metrics.Max_Preciptation:F2};" +
                             $"{metrics.Level:F2}");
        }
        Console.WriteLine($"Data written to {outputFilePath}");
    }

    public static float CalculateAverageExcludingMinMax(IEnumerable<float> values)
    {
        var valueList = values.ToList();

        if (valueList.Count <= 2) return 0;

        float min = valueList.Min();
        float max = valueList.Max();

        int minCount = valueList.Count(v => v == min);
        int maxCount = valueList.Count(v => v == max);

        float sum = valueList.Sum(v => v);
        float adjustedSum = sum - (min * minCount) - (max * maxCount);
        int adjustedCount = valueList.Count - (minCount + maxCount);

        return adjustedCount > 0 ? adjustedSum / adjustedCount : 0;
    }

    public static List<WeatherData> ReadWeatherData(string filePath)
    {
        List<WeatherData> weatherDataList = [];
        using StreamReader reader = new(filePath);
        _ = reader.ReadLine();
        while (!reader.EndOfStream)
        {
            string? line = reader.ReadLine();
            string[]? values = line?.Split(';');
            if (values?.Length == 6)
                weatherDataList.Add(new()
                {
                    Data = DateTime.Parse(values[0]),
                    Hora = TimeSpan.Parse(values[1]),
                    Chuva = float.Parse(values[2]),
                    Nivel = float.Parse(values[3]),
                    Temperatura = float.Parse(values[4]),
                    Umidade = float.Parse(values[5])
                });
        }

        return weatherDataList;
    }
}
