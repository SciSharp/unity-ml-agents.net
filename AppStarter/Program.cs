using System;
using UnitTests;

namespace AppStarter
{
    class Program
    {
        static void Main(string[] args)
        {
            var tests = new test_ppo();
            tests.test_ppo_model_dc_vector();

            Console.ReadLine();
        }
    }
}
